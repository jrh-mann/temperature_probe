#!/usr/bin/env python3
"""
Train linear probes on activations for all layers.

Usage:
    python train_probe.py --activations_dir /workspace/activations/Qwen3-0.6B --token_every_n 10
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import gc

from utils.activations import load_activations


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_linear_probe(X_train, y_train, X_val, y_val, device='cuda', epochs=1000, lr=0.01, patience=50):
    """Train linear probe with early stopping."""
    model = LinearProbe(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = nn.functional.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.functional.mse_loss(val_pred, y_val).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_model_state is not None:
        best_model_state = {k: v.to(device) for k, v in best_model_state.items()}
        model.load_state_dict(best_model_state)
    
    return model


def compute_metrics(y_true, y_pred):
    """Compute RMSE and R²."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return rmse, r2


def load_layers_batch(activations_dir, layer_indices, token_every_n, chunk_size=10, max_workers=4):
    """
    Load activations from all temperature directories for specified layers at once.
    This avoids re-reading files for each layer.
    Returns a dictionary mapping layer_idx -> (X, Y) for specified layers.
    
    Args:
        layer_indices: List of layer indices to load (e.g., [0, 1, 2, 3])
        chunk_size: Number of files to concatenate into each intermediate chunk
        max_workers: Number of parallel workers for file loading
    """
    activations_dir = Path(activations_dir)
    
    # Find all temperature directories
    temp_dirs = sorted([d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith('temperature_')])
    
    if not temp_dirs:
        raise ValueError(f"No temperature directories found in {activations_dir}")
    
    print(f"Found {len(temp_dirs)} temperature directories")
    print(f"Loading layers {layer_indices} at once with {max_workers} parallel workers...")
    
    # Storage for specified layers: layer_idx -> list of chunks
    layer_data = {i: {'X_chunks': [], 'Y_chunks': [], 'current_chunk_X': [], 'current_chunk_Y': [], 'files_in_chunk': 0, 'total_samples': 0}
                  for i in layer_indices}
    
    # Process each temperature directory
    for temp_dir in temp_dirs:
        # Extract temperature from directory name
        try:
            temp = float(temp_dir.name.replace('temperature_', ''))
        except ValueError:
            print(f"Warning: Could not parse temperature from {temp_dir.name}, skipping")
            continue
        
        # Load ONLY the requested layers (memory efficient!)
        try:
            # Load only the requested layers from all files in this temperature directory
            result = load_activations(str(temp_dir), layer_indices=layer_indices, token_every_n=token_every_n, max_workers=max_workers)
            act_list, extracted_layer_indices = result
            
            # Create mapping: layer_idx -> position in acts tensor
            # extracted_layer_indices is sorted list of layers that exist
            layer_to_pos = {layer_idx: i for i, layer_idx in enumerate(extracted_layer_indices)}
            
            # Process each file's activations (contains only requested layers that exist)
            for acts in act_list:
                # acts shape: (len(extracted_layer_indices), seq_len, hidden_dim)
                if acts.shape[0] == 0 or acts.numel() == 0:
                    # Empty tensor (layers didn't exist in this file)
                    continue
                
                seq_len = acts.shape[1]
                
                # Extract each requested layer that exists in this file
                for layer_idx in layer_indices:
                    if layer_idx not in layer_to_pos:
                        # Layer doesn't exist in any file (shouldn't happen if detection worked)
                        continue
                    
                    pos = layer_to_pos[layer_idx]
                    if pos >= acts.shape[0]:
                        # This specific file doesn't have this layer (can happen if files vary)
                        continue
                    
                    layer_acts = acts[pos].clone()  # (seq_len, hidden_dim) - clone to break reference
                    
                    layer_data[layer_idx]['current_chunk_X'].append(layer_acts)
                    layer_data[layer_idx]['current_chunk_Y'].append(torch.full((seq_len,), temp, dtype=torch.float32))
                    layer_data[layer_idx]['files_in_chunk'] += 1
                    layer_data[layer_idx]['total_samples'] += seq_len
                    
                    # When chunk is full, concatenate and add to chunks list
                    if layer_data[layer_idx]['files_in_chunk'] >= chunk_size:
                        X_chunk = torch.cat(layer_data[layer_idx]['current_chunk_X'], dim=0)
                        Y_chunk = torch.cat(layer_data[layer_idx]['current_chunk_Y'], dim=0)
                        layer_data[layer_idx]['X_chunks'].append(X_chunk)
                        layer_data[layer_idx]['Y_chunks'].append(Y_chunk)
                        
                        # Clean up individual tensors in the chunk
                        del layer_data[layer_idx]['current_chunk_X'][:]
                        del layer_data[layer_idx]['current_chunk_Y'][:]
                        layer_data[layer_idx]['current_chunk_X'] = []
                        layer_data[layer_idx]['current_chunk_Y'] = []
                        layer_data[layer_idx]['files_in_chunk'] = 0
                        gc.collect()
                
                # Clean up the activation tensor (now only contains requested layers)
                del acts
                gc.collect()
        
            # Clean up the list
            del act_list
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Error loading {temp_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Finalize specified layers: concatenate remaining chunks and create final tensors
    print(f"Finalizing activations for layers {layer_indices}...")
    layer_activations = {}
    
    for layer_idx in layer_indices:
        # Concatenate any remaining files in the current chunk
        if layer_data[layer_idx]['current_chunk_X']:
            X_chunk = torch.cat(layer_data[layer_idx]['current_chunk_X'], dim=0)
            Y_chunk = torch.cat(layer_data[layer_idx]['current_chunk_Y'], dim=0)
            layer_data[layer_idx]['X_chunks'].append(X_chunk)
            layer_data[layer_idx]['Y_chunks'].append(Y_chunk)
        
        if not layer_data[layer_idx]['X_chunks']:
            print(f"Warning: No activations loaded for layer {layer_idx}")
            continue
        
        # Final concatenation
        X = torch.cat(layer_data[layer_idx]['X_chunks'], dim=0)
        Y = torch.cat(layer_data[layer_idx]['Y_chunks'], dim=0)
        
        layer_activations[layer_idx] = (X, Y)
        
        print(f"Layer {layer_idx}: X shape {X.shape}, Y shape {Y.shape}")
        
        # Clean up
        del layer_data[layer_idx]
        gc.collect()
    
    return layer_activations


def load_all_activations(activations_dir, layer_idx, token_every_n, device='cuda', chunk_size=50, max_workers=8):
    """
    Load activations from all temperature directories for a specific layer.
    (Legacy function for backward compatibility - consider using load_all_layers_at_once)
    """
    activations_dir = Path(activations_dir)
    
    # Find all temperature directories
    temp_dirs = sorted([d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith('temperature_')])
    
    if not temp_dirs:
        raise ValueError(f"No temperature directories found in {activations_dir}")
    
    print(f"Found {len(temp_dirs)} temperature directories")
    
    # Accumulate into larger chunks to reduce final concatenation overhead
    X_chunks = []
    Y_chunks = []
    total_samples = 0
    
    # Temporary buffer for accumulating files into chunks
    current_chunk_X = []
    current_chunk_Y = []
    files_in_chunk = 0
    
    for temp_dir in temp_dirs:
        # Extract temperature from directory name
        try:
            temp = float(temp_dir.name.replace('temperature_', ''))
        except ValueError:
            print(f"Warning: Could not parse temperature from {temp_dir.name}, skipping")
            continue
        
        # Load activations for this layer
        try:
            act_list = load_activations(str(temp_dir), layer_idx=layer_idx, token_every_n=token_every_n, max_workers=max_workers)
            
            # Process each file and accumulate into chunks
            for acts in act_list:
                # acts shape: (seq_len, hidden_dim)
                seq_len = acts.shape[0]
                
                current_chunk_X.append(acts)
                current_chunk_Y.append(torch.full((seq_len,), temp, dtype=torch.float32))
                files_in_chunk += 1
                total_samples += seq_len
                
                # When chunk is full, concatenate and add to chunks list
                if files_in_chunk >= chunk_size:
                    # Concatenate this chunk
                    X_chunk = torch.cat(current_chunk_X, dim=0)
                    Y_chunk = torch.cat(current_chunk_Y, dim=0)
                    X_chunks.append(X_chunk)
                    Y_chunks.append(Y_chunk)
                    
                    # Clean up
                    del current_chunk_X, current_chunk_Y
                    current_chunk_X = []
                    current_chunk_Y = []
                    files_in_chunk = 0
                    gc.collect()
            
            # Clean up the list
            del act_list
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Error loading {temp_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Concatenate any remaining files in the current chunk
    if current_chunk_X:
        X_chunk = torch.cat(current_chunk_X, dim=0)
        Y_chunk = torch.cat(current_chunk_Y, dim=0)
        X_chunks.append(X_chunk)
        Y_chunks.append(Y_chunk)
        del current_chunk_X, current_chunk_Y
        gc.collect()
    
    if not X_chunks:
        raise ValueError("No activations loaded!")
    
    print(f"Concatenating {len(X_chunks)} chunks ({total_samples} total samples)...")
    
    # Final concatenation (now with fewer, larger chunks)
    X = torch.cat(X_chunks, dim=0)
    Y = torch.cat(Y_chunks, dim=0)
    
    # Clean up chunks immediately
    del X_chunks, Y_chunks
    gc.collect()
    
    print(f"Loaded activations: X shape {X.shape}, Y shape {Y.shape}")
    
    # Move to device only at the end
    if device != 'cpu' and torch.cuda.is_available():
        print(f"Moving to {device}...")
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        torch.cuda.empty_cache()
    
    return X, Y


def main():
    parser = argparse.ArgumentParser(description='Train linear probes on activations')
    parser.add_argument('--activations_dir', type=str, required=True,
                        help='Path to activations directory (e.g., /workspace/activations/Qwen3-0.6B)')
    parser.add_argument('--token_every_n', type=int, default=10,
                        help='Sample every nth token (default: 10 = every 10th token, reduces memory significantly)')
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='Number of files to concatenate into each intermediate chunk (default: 10, increase for larger chunks)')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Number of parallel workers for loading files (default: 4, increase for faster loading)')
    parser.add_argument('--layers_per_batch', type=int, default=4,
                        help='Number of layers to load and process in each batch (default: 4)')
    parser.add_argument('--max_files_per_temp', type=int, default=None,
                        help='Maximum number of files to load per temperature (default: None = all files). Useful for faster testing.')
    parser.add_argument('--train_classifier', action='store_true',
                        help='Also train a classifier to predict temperature class (in addition to regression)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum training epochs (default: 1000)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if args.token_every_n == 1:
        print("WARNING: token_every_n=1 loads ALL tokens. This may cause OOM.")
        print("Consider using --token_every_n 10 or higher to reduce memory usage.")
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # First, determine number of layers by loading one file
    activations_dir = Path(args.activations_dir)
    temp_dirs = sorted([d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith('temperature_')])
    if not temp_dirs:
        raise ValueError(f"No temperature directories found in {activations_dir}")
    
    # Check activation files to get number of layers
    # Activation files are (layers, 1, seq_len, d_model)
    # Sample a few files to find the minimum number of layers (some files might have fewer)
    first_temp_dir = temp_dirs[0]
    act_files = sorted([f for f in os.listdir(first_temp_dir) if f.endswith('.pt') and f != 'prompts.pt'])
    if not act_files:
        raise ValueError(f"No activation files found in {first_temp_dir}")
    
    # Check first few files to determine number of layers
    num_layers_list = []
    for act_file in act_files[:min(5, len(act_files))]:  # Check first 5 files
        sample_acts = torch.load(first_temp_dir / act_file, map_location='cpu')
        # sample_acts shape: (num_layers, 1, seq_len, hidden_dim)
        num_layers_list.append(sample_acts.shape[0])
        del sample_acts
    
    # Use minimum to be safe (some files might have fewer layers)
    num_layers = min(num_layers_list)
    max_layers = max(num_layers_list)
    if num_layers != max_layers:
        print(f"Warning: Files have inconsistent layer counts (min: {num_layers}, max: {max_layers})")
        print(f"Using {num_layers} layers (minimum) to ensure compatibility")
    else:
        print(f"Detected {num_layers} layers in all files")
    gc.collect()
    
    # Storage for results across all layers
    all_results = {
        'layer': [],
        'train_rmse': [],
        'test_rmse': [],
        'test_r2': []
    }
    
    # Process layers in batches
    num_batches = (num_layers + args.layers_per_batch - 1) // args.layers_per_batch
    print(f"\nProcessing {num_layers} layers in {num_batches} batches of {args.layers_per_batch} layers each")
    
    for batch_idx in range(num_batches):
        start_layer = batch_idx * args.layers_per_batch
        end_layer = min(start_layer + args.layers_per_batch, num_layers)
        layer_batch = list(range(start_layer, end_layer))
        
        print(f"\n{'='*80}")
        print(f"Batch {batch_idx + 1}/{num_batches}: Loading layers {layer_batch}")
        print(f"{'='*80}")
        
        # Load this batch of layers
        layer_activations = load_layers_batch(
            args.activations_dir, layer_batch, args.token_every_n,
            chunk_size=args.chunk_size, max_workers=args.max_workers
        )
        
        # Train probe for each layer in this batch
        for layer_idx in layer_batch:
            print(f"\n{'='*80}")
            print(f"Layer {layer_idx}/{num_layers-1}")
            print(f"{'='*80}")
            
            try:
                # Get activations for this layer (already loaded)
                if layer_idx not in layer_activations:
                    print(f"Warning: No activations found for layer {layer_idx}, skipping")
                    continue
                
                X, Y = layer_activations[layer_idx]
                
                # Split into train/test (keep on CPU for now)
                X_cpu = X.cpu() if X.is_cuda else X
                Y_cpu = Y.cpu() if Y.is_cuda else Y
                
                indices = np.arange(len(X_cpu))
                Y_np = Y_cpu.numpy()
                
                # Create stratification labels from temperature values (discrete: 0, 0.5, 1.0, 1.5)
                temp_labels = Y_np.astype(str)  # Convert to string for stratification
                train_indices, test_indices = train_test_split(
                    indices, test_size=args.test_size, random_state=42, stratify=temp_labels
                )
                
                # Split on CPU first
                X_train = X_cpu[train_indices]
                X_test = X_cpu[test_indices]
                y_train = Y_cpu[train_indices]
                y_test = Y_cpu[test_indices]
                
                # Clean up full tensors
                del X, Y, X_cpu, Y_cpu
                gc.collect()
                
                # Further split train into train/val (use relative indices)
                val_size = int(0.2 * len(train_indices))
                train_subset_indices = np.arange(len(train_indices))
                np.random.seed(42)
                np.random.shuffle(train_subset_indices)
                val_subset_indices = train_subset_indices[:val_size]
                train_final_subset_indices = train_subset_indices[val_size:]
                
                X_train_final = X_train[train_final_subset_indices]
                X_val = X_train[val_subset_indices]
                y_train_final = y_train[train_final_subset_indices]
                y_val = y_train[val_subset_indices]
                
                # Clean up intermediate tensors
                del X_train, y_train
                gc.collect()
                
                # Move to device only the final splits
                if args.device != 'cpu' and torch.cuda.is_available():
                    device_torch = torch.device(args.device)
                    X_train_final = X_train_final.to(device_torch, non_blocking=True)
                    X_val = X_val.to(device_torch, non_blocking=True)
                    X_test = X_test.to(device_torch, non_blocking=True)
                    y_train_final = y_train_final.to(device_torch, non_blocking=True)
                    y_val = y_val.to(device_torch, non_blocking=True)
                    y_test = y_test.to(device_torch, non_blocking=True)
                    torch.cuda.empty_cache()
                
                # Standardize
                X_mean = X_train_final.mean(dim=0, keepdim=True)
                X_std = X_train_final.std(dim=0, keepdim=True) + 1e-8
                X_train_scaled = (X_train_final - X_mean) / X_std
                X_val_scaled = (X_val - X_mean) / X_std
                X_test_scaled = (X_test - X_mean) / X_std
                
                print(f"Training data: {len(X_train_final)} samples")
                print(f"Validation data: {len(X_val)} samples")
                print(f"Test data: {len(X_test)} samples")
                
                # Train model
                print("Training linear probe...")
                device_for_training = torch.device(args.device) if args.device != 'cpu' and torch.cuda.is_available() else torch.device('cpu')
                model = train_linear_probe(
                    X_train_scaled, y_train_final, X_val_scaled, y_val,
                    device=device_for_training, epochs=args.epochs, lr=args.lr, patience=args.patience
                )
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    y_pred_train = model(X_train_scaled).cpu().numpy()
                    y_pred_test = model(X_test_scaled).cpu().numpy()
                
                y_train_np = y_train_final.cpu().numpy() if y_train_final.is_cuda else y_train_final.numpy()
                y_test_np = y_test.cpu().numpy() if y_test.is_cuda else y_test.numpy()
                
                train_rmse, train_r2 = compute_metrics(y_train_np, y_pred_train)
                test_rmse, test_r2 = compute_metrics(y_test_np, y_pred_test)
                
                # Print metrics
                print(f"\nLayer {layer_idx} Results:")
                print(f"  Train RMSE: {train_rmse:.6f}")
                print(f"  Train R²:   {train_r2:.6f}")
                print(f"  Test RMSE:  {test_rmse:.6f}")
                print(f"  Test R²:    {test_r2:.6f}")
                
                # Save results (convert to Python native types for JSON serialization)
                all_results['layer'].append(int(layer_idx))
                all_results['train_rmse'].append(float(train_rmse))
                all_results['test_rmse'].append(float(test_rmse))
                all_results['test_r2'].append(float(test_r2))
                
                # Save model
                model_path = Path('models') / f'layer_{layer_idx}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'X_mean': X_mean.cpu(),
                    'X_std': X_std.cpu(),
                    'layer': layer_idx,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2
                }, model_path)
                
                # Clean up memory aggressively
                del X_train_final, X_val, X_test, y_train_final, y_val, y_test
                del X_train_scaled, X_val_scaled, X_test_scaled
                del model, y_pred_train, y_pred_test, y_train_np, y_test_np
                if args.device != 'cpu' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error processing layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Clean up this batch of layer activations before loading next batch
        print(f"\nCleaning up batch {batch_idx + 1} activations...")
        del layer_activations
        gc.collect()
        if args.device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create plots
    print(f"\n{'='*80}")
    print("Creating plots...")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    layers = all_results['layer']
    train_rmse = all_results['train_rmse']
    test_rmse = all_results['test_rmse']
    test_r2 = all_results['test_r2']
    
    # Plot 1: RMSE
    ax1 = axes[0]
    ax1.plot(layers, train_rmse, 'o-', label='Train RMSE', linewidth=2, markersize=4)
    ax1.plot(layers, test_rmse, 's-', label='Test RMSE', linewidth=2, markersize=4)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Linear Probe Performance: RMSE vs Layer', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    best_layer_rmse = layers[np.argmin(test_rmse)]
    ax1.axvline(x=best_layer_rmse, color='red', linestyle='--', alpha=0.5, label=f'Best (layer {best_layer_rmse})')
    ax1.legend(fontsize=11)
    
    # Plot 2: R²
    ax2 = axes[1]
    ax2.plot(layers, test_r2, '^-', label='Test R²', linewidth=2, markersize=4, color='green')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('Linear Probe Performance: Test R² vs Layer', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    best_layer_r2 = layers[np.argmax(test_r2)]
    ax2.axvline(x=best_layer_r2, color='red', linestyle='--', alpha=0.5, label=f'Best (layer {best_layer_r2})')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plot_path = Path('plots') / 'layer_probe_performance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()
    
    # Save results to JSON
    results_path = Path('plots') / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    best_rmse_idx = np.argmin(test_rmse)
    best_r2_idx = np.argmax(test_r2)
    print(f"Best layer (by Test RMSE): Layer {layers[best_rmse_idx]}")
    print(f"  Test RMSE: {test_rmse[best_rmse_idx]:.6f}")
    print(f"  Test R²:   {test_r2[best_rmse_idx]:.6f}")
    print()
    print(f"Best layer (by Test R²): Layer {layers[best_r2_idx]}")
    print(f"  Test RMSE: {test_rmse[best_r2_idx]:.6f}")
    print(f"  Test R²:   {test_r2[best_r2_idx]:.6f}")
    print()
    print(f"Average Test RMSE: {np.mean(test_rmse):.6f} ± {np.std(test_rmse):.6f}")
    print(f"Average Test R²: {np.mean(test_r2):.6f} ± {np.std(test_r2):.6f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

