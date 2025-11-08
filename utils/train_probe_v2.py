#!/usr/bin/env python3
"""
Temperature Probe Training Script (V2)
- Fixed data leakage: splits by files, not tokens
- Trains both regression and classification probes
- Memory efficient with batch loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import json
import gc
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import memory profiling tools
try:
    import ctypes
    HAS_CTYPES = True
except ImportError:
    HAS_CTYPES = False


def aggressive_gc():
    """Aggressively free memory."""
    # Run garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # On Linux, try to force malloc to release memory to OS
    if HAS_CTYPES and sys.platform == 'linux':
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass


class LinearProbe(nn.Module):
    """Simple linear probe for regression."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)


class ClassificationProbe(nn.Module):
    """Linear probe for classification."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


def load_file_activations(file_path, layer_indices, sample_tokens_per_sequence=50):
    """
    Load activations from a single file for specified layers.
    
    Args:
        file_path: Path to .pt file
        layer_indices: List of layer indices to extract
        sample_tokens_per_sequence: If > 0, uniformly sample up to this many tokens per sequence
    
    Returns:
        Dict mapping layer_idx -> tensor of shape (seq_len, hidden_dim)
    """
    acts = torch.load(file_path, map_location='cpu')
    if isinstance(acts, torch.Tensor) and acts.dtype != torch.float32:
        acts = acts.to(dtype=torch.float32)
    # acts shape: (num_layers, 1, seq_len, hidden_dim)
    
    result = {}
    for layer_idx in layer_indices:
        if layer_idx >= acts.shape[0]:
            continue
        
        # Extract layer and remove batch dim: (seq_len, hidden_dim)
        layer_acts = acts[layer_idx].squeeze(0)
        if layer_acts.dtype != torch.float32:
            layer_acts = layer_acts.to(dtype=torch.float32)
        
        seq_len = layer_acts.shape[0]
        if sample_tokens_per_sequence and sample_tokens_per_sequence > 0 and seq_len > sample_tokens_per_sequence:
            indices = torch.linspace(
                0,
                seq_len - 1,
                steps=sample_tokens_per_sequence,
                dtype=torch.float32
            ).round().to(dtype=torch.long)
            indices = torch.unique(indices, sorted=True)
            if indices.numel() > sample_tokens_per_sequence:
                indices = indices[:sample_tokens_per_sequence]
            layer_acts = layer_acts[indices]
        
        result[layer_idx] = layer_acts
    
    del acts
    gc.collect()
    
    return result


def load_data_by_files(
    activations_dir,
    layer_indices,
    sample_tokens_per_sequence=50,
    max_files_per_temp=None,
    test_size=0.1,
    val_size=0.1,
    preload_device='cpu',
    files_per_batch=4,
    chunk_size=8,
):
    """
    Load activations and split by files (not tokens) to avoid data leakage.
    Can load multiple layers at once for efficiency.
    
    Args:
        layer_indices: List of layer indices to load (e.g., [0, 1, 2, 3])
        preload_device: Device to preload data to ('cpu', 'cuda', or 'auto')
                       'auto' will use GPU if available and there's space
        files_per_batch: Number of activation files to process before forcing a cleanup step.
        chunk_size: Number of files to accumulate per layer before concatenating into a larger tensor.
        sample_tokens_per_sequence: Max tokens to sample per file/sequence (uniform spacing). Use <=0 to keep all tokens.
    
    Returns:
        Dictionary mapping layer_idx -> (train_data, val_data, test_data)
        temp_to_class: Dict mapping temperature to class index
        class_to_temp: Dict mapping class index to temperature
    """
    activations_dir = Path(activations_dir)
    
    if files_per_batch < 1:
        raise ValueError("files_per_batch must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    
    # Find all temperature directories
    temp_dirs = sorted([d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith('temperature_')])
    
    if not temp_dirs:
        raise ValueError(f"No temperature directories found in {activations_dir}")
    
    print(f"Found {len(temp_dirs)} temperature directories")
    
    # Create temperature to class mapping
    temperatures = sorted([float(d.name.replace('temperature_', '')) for d in temp_dirs])
    temp_to_class = {temp: i for i, temp in enumerate(temperatures)}
    class_to_temp = {i: temp for temp, i in temp_to_class.items()}
    
    print(f"Temperature classes: {temperatures}")
    
    # Collect files per temperature
    files_by_temp = {}
    for temp_dir in temp_dirs:
        temp = float(temp_dir.name.replace('temperature_', ''))
        act_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.pt') and f != 'prompts.pt'])
        
        # Limit files if requested
        if max_files_per_temp is not None:
            act_files = act_files[:max_files_per_temp]
        
        files_by_temp[temp] = [(temp_dir / f) for f in act_files]
        print(f"  Temperature {temp}: {len(files_by_temp[temp])} files")
    
    # Split files into train/val/test for each temperature (stratified by temperature)
    train_files = []
    val_files = []
    test_files = []
    
    for temp, files in files_by_temp.items():
        if len(files) < 3:
            print(f"Warning: Only {len(files)} files for temp {temp}, using all for training")
            train_files.extend([(f, temp) for f in files])
            continue
        
        # First split: train+val vs test
        files_train_val, files_test = train_test_split(
            files, test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        if len(files_train_val) > 1:
            files_train, files_val = train_test_split(
                files_train_val, test_size=val_size/(1-test_size), random_state=42
            )
        else:
            files_train = files_train_val
            files_val = []
        
        train_files.extend([(f, temp) for f in files_train])
        val_files.extend([(f, temp) for f in files_val])
        test_files.extend([(f, temp) for f in files_test])
    
    print(f"\nFile splits:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")
    
    # Load activations for each split and each layer
    def load_split(file_list, split_name, move_to_device=None):
        layer_batches = {layer_idx: [] for layer_idx in layer_indices}
        label_batches = {layer_idx: [] for layer_idx in layer_indices}
        layer_chunks = {layer_idx: [] for layer_idx in layer_indices}
        label_chunks = {layer_idx: [] for layer_idx in layer_indices}
        layer_chunk_counts = {layer_idx: 0 for layer_idx in layer_indices}
        
        target_device = move_to_device
        gpu_enabled = target_device is not None and target_device.type == 'cuda'
        
        def move_chunks_to_cpu():
            for idx in layer_indices:
                if layer_chunks[idx]:
                    layer_chunks[idx] = [chunk.to('cpu', dtype=torch.float32) for chunk in layer_chunks[idx]]
                if label_chunks[idx]:
                    label_chunks[idx] = [labels.to('cpu', dtype=torch.float32) for labels in label_chunks[idx]]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        def flush_layer(layer_idx):
            nonlocal gpu_enabled
            if not layer_chunks[layer_idx]:
                return
            
            X_chunk = torch.cat(layer_chunks[layer_idx], dim=0)
            Y_chunk = torch.cat(label_chunks[layer_idx], dim=0)
            layer_chunks[layer_idx].clear()
            label_chunks[layer_idx].clear()
            layer_chunk_counts[layer_idx] = 0
            
            if gpu_enabled:
                try:
                    X_chunk = X_chunk.to(target_device, dtype=torch.float32, non_blocking=True)
                    Y_chunk = Y_chunk.to(target_device, non_blocking=True)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("⚠️ GPU OOM while preloading, falling back to CPU for remaining data.")
                        gpu_enabled = False
                        move_chunks_to_cpu()
                        X_chunk = X_chunk.to('cpu', dtype=torch.float32)
                        Y_chunk = Y_chunk.to('cpu', dtype=torch.float32)
                    else:
                        raise
            else:
                X_chunk = X_chunk.to('cpu', dtype=torch.float32)
                Y_chunk = Y_chunk.to('cpu', dtype=torch.float32)
            
            layer_batches[layer_idx].append(X_chunk)
            label_batches[layer_idx].append(Y_chunk)
            aggressive_gc()
        
        def flush_all_layers():
            for idx in layer_indices:
                flush_layer(idx)
        
        total_steps = (len(file_list) + files_per_batch - 1) // files_per_batch
        for batch_start in tqdm(range(0, len(file_list), files_per_batch), desc=f"Loading {split_name}", total=total_steps):
            batch = file_list[batch_start:batch_start + files_per_batch]
            for file_path, temp in batch:
                acts_dict = load_file_activations(
                    file_path,
                    layer_indices,
                    sample_tokens_per_sequence=sample_tokens_per_sequence,
                )
                
                for layer_idx, acts in acts_dict.items():
                    if acts is None or acts.shape[0] == 0:
                        continue
                    
                    if acts.dtype != torch.float32:
                        acts = acts.to(dtype=torch.float32)
                    
                    if gpu_enabled:
                        try:
                            acts = acts.to(target_device, dtype=torch.float32, non_blocking=True)
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print("⚠️ GPU OOM while preloading, falling back to CPU for remaining data.")
                                gpu_enabled = False
                                move_chunks_to_cpu()
                                acts = acts.to('cpu', dtype=torch.float32)
                            else:
                                raise
                    else:
                        acts = acts.to('cpu', dtype=torch.float32)
                    
                    labels = torch.full(
                        (acts.shape[0],),
                        temp,
                        dtype=torch.float32,
                        device=acts.device
                    )
                    
                    layer_chunks[layer_idx].append(acts)
                    label_chunks[layer_idx].append(labels)
                    layer_chunk_counts[layer_idx] += 1
                    
                    if layer_chunk_counts[layer_idx] >= chunk_size:
                        flush_layer(layer_idx)
                
                del acts_dict
                aggressive_gc()
            
            aggressive_gc()
        
        flush_all_layers()
        
        result = {}
        for layer_idx in layer_indices:
            if layer_batches[layer_idx]:
                X = torch.cat(layer_batches[layer_idx], dim=0)
                Y = torch.cat(label_batches[layer_idx], dim=0)
                result[layer_idx] = (X, Y)
            else:
                result[layer_idx] = (None, None)
        
        layer_batches.clear()
        label_batches.clear()
        layer_chunks.clear()
        label_chunks.clear()
        aggressive_gc()
        
        return result
    
    # Determine target device for preloading
    if preload_device == 'auto':
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        target_device = torch.device(preload_device)
    
    # Determine if we should move to GPU
    move_to_device = target_device if target_device.type == 'cuda' else None
    
    if move_to_device:
        print(f"\nStreaming data to GPU to minimize RAM usage...")
        if torch.cuda.is_available():
            print(f"GPU Memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load and immediately transfer to GPU for each split
    # This keeps RAM usage low by freeing CPU memory as we go
    print("\nLoading train split...")
    train_dict = load_split(train_files, "train", move_to_device=move_to_device)
    print(f"✓ Train data loaded{' and transferred to GPU' if move_to_device else ''}")
    aggressive_gc()
    
    if move_to_device and torch.cuda.is_available():
        print(f"GPU Memory after train: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    if val_files:
        print("\nLoading val split...")
        val_dict = load_split(val_files, "val", move_to_device=move_to_device)
        print(f"✓ Val data loaded{' and transferred to GPU' if move_to_device else ''}")
        aggressive_gc()
        
        if move_to_device and torch.cuda.is_available():
            print(f"GPU Memory after val: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        val_dict = {l: (None, None) for l in layer_indices}
    
    print("\nLoading test split...")
    test_dict = load_split(test_files, "test", move_to_device=move_to_device)
    print(f"✓ Test data loaded{' and transferred to GPU' if move_to_device else ''}")
    aggressive_gc()
    
    if move_to_device and torch.cuda.is_available():
        print(f"GPU Memory after test: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Organize data by layer
    data_by_layer = {}
    
    for layer_idx in layer_indices:
        X_train, y_train = train_dict[layer_idx]
        X_val, y_val = val_dict[layer_idx]
        X_test, y_test = test_dict[layer_idx]
        
        if X_train is not None:
            print(f"\nLayer {layer_idx} data shapes:")
            print(f"  Train: X={X_train.shape}, y={y_train.shape} [{X_train.device}]")
            if X_val is not None:
                print(f"  Val: X={X_val.shape}, y={y_val.shape} [{X_val.device}]")
            if X_test is not None:
                print(f"  Test: X={X_test.shape}, y={y_test.shape} [{X_test.device}]")
            
            data_by_layer[layer_idx] = {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }
    
    if move_to_device and torch.cuda.is_available():
        print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Cleanup dicts aggressively
    del train_dict, val_dict, test_dict
    aggressive_gc()
    
    return data_by_layer, temp_to_class, class_to_temp


def train_regression_probe(X_train, y_train, X_val, y_val, device, epochs=1000, lr=0.001, patience=50):
    """Train linear regression probe."""
    input_dim = X_train.shape[1]
    model = LinearProbe(input_dim).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Standardize (compute in float32 for numerical stability)
    X_train_float = X_train.to(dtype=torch.float32)
    X_mean = X_train_float.mean(dim=0, keepdim=True)
    X_std = X_train_float.std(dim=0, keepdim=True) + 1e-8
    
    # Standardize and ensure on correct device
    X_mean_bf16 = X_mean.to(device=X_train.device, dtype=X_train.dtype)
    X_std_bf16 = X_std.to(device=X_train.device, dtype=X_train.dtype)
    X_train_scaled = (X_train - X_mean_bf16) / X_std_bf16
    X_train_scaled = X_train_scaled.to(device, dtype=torch.float32, non_blocking=True)
    
    y_train_float = y_train if y_train.device == device else y_train.to(device, non_blocking=True)
    y_train_float = y_train_float.to(dtype=torch.float32)
    
    if X_val is not None:
        X_val_scaled = (X_val - X_mean.to(device=X_val.device, dtype=X_val.dtype)) / X_std.to(device=X_val.device, dtype=X_val.dtype)
        X_val_scaled = X_val_scaled.to(device, dtype=torch.float32, non_blocking=True)
        y_val_float = y_val if y_val.device == device else y_val.to(device, non_blocking=True)
        y_val_float = y_val_float.to(dtype=torch.float32)
    else:
        X_val_scaled = None
        y_val_t = None
        y_val_float = None
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        y_pred = model(X_train_scaled)
        if epoch == 0:
            print("  Epoch 1 regression preds sample (first 5):")
            print(y_pred[:5].detach().cpu())
        loss = criterion(y_pred.to(dtype=torch.float32), y_train_float)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        if X_val is not None and X_val_scaled is not None and y_val_float is not None:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_scaled)
                if epoch == 0:
                    print("  Epoch 1 regression val preds sample (first 5):")
                    print(y_val_pred[:5].detach().cpu())
                val_loss = criterion(y_val_pred.to(dtype=torch.float32), y_val_float).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, X_mean, X_std


def train_classification_probe(X_train, y_train, X_val, y_val, temp_to_class, class_to_temp, device, epochs=1000, lr=0.001, patience=50):
    """Train linear classification probe."""
    input_dim = X_train.shape[1]
    num_classes = len(temp_to_class)
    model = ClassificationProbe(input_dim, num_classes).to(device=device, dtype=torch.float32)
    
    # Compute class weights to counter temperature imbalance
    y_train_cpu = y_train.cpu() if y_train.is_cuda else y_train
    y_train_classes_list = [temp_to_class[val.item()] for val in y_train_cpu]
    class_counts = torch.bincount(torch.tensor(y_train_classes_list, dtype=torch.long), minlength=num_classes).float()
    class_weights = class_counts.sum() / (class_counts + 1e-8)
    class_weights = class_weights.to(device)
    print("\n  Class counts (train):")
    for idx, temp in class_to_temp.items():
        print(f"    Class {idx} (temp {temp}): {int(class_counts[idx].item())} samples, weight={class_weights[idx].item():.4f}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Standardize (compute in float32 for numerical stability)
    X_train_float = X_train.to(dtype=torch.float32)
    X_mean = X_train_float.mean(dim=0, keepdim=True)
    X_std = X_train_float.std(dim=0, keepdim=True) + 1e-8
    
    X_train_scaled = (X_train - X_mean.to(device=X_train.device, dtype=X_train.dtype)) / X_std.to(device=X_train.device, dtype=X_train.dtype)
    X_train_scaled = X_train_scaled.to(device, dtype=torch.float32, non_blocking=True)
    
    # Convert temps to class labels (handle both CPU and GPU tensors)
    y_train_classes = torch.tensor(y_train_classes_list, dtype=torch.long).to(device)
    
    if X_val is not None:
        X_val_scaled = (X_val - X_mean.to(device=X_val.device, dtype=X_val.dtype)) / X_std.to(device=X_val.device, dtype=X_val.dtype)
        X_val_scaled = X_val_scaled.to(device, dtype=torch.float32, non_blocking=True)
        y_val_cpu = y_val.cpu() if y_val.is_cuda else y_val
        y_val_classes = torch.tensor([temp_to_class[y.item()] for y in y_val_cpu], dtype=torch.long).to(device)
        print(f"  Validation class counts: {[int((y_val_cpu == temp).sum().item()) for temp in sorted(temp_to_class.keys())]}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_train_scaled)
        if epoch == 0:
            print("  Epoch 1 logits sample (first 5):")
            print(logits[:5].detach().cpu())
        loss = criterion(logits.to(dtype=torch.float32), y_train_classes)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val_scaled)
                if epoch == 0:
                    print("  Epoch 1 val logits sample (first 5):")
                    print(logits_val[:5].detach().cpu())
                val_loss = criterion(logits_val.to(dtype=torch.float32), y_val_classes).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, X_mean, X_std


def evaluate_regression(model, X, y, X_mean, X_std, device):
    """Evaluate regression probe."""
    model.eval()
    
    # Move X to device for model inference if needed
    X_scaled = (X - X_mean.to(device=X.device, dtype=X.dtype)) / X_std.to(device=X.device, dtype=X.dtype)
    X_scaled = X_scaled.to(device, dtype=torch.float32, non_blocking=True)
    
    with torch.no_grad():
        y_pred = model(X_scaled).to(dtype=torch.float32).cpu().numpy()
    
    # Handle both CPU and CUDA tensors
    y_cpu = y.detach().cpu() if y.is_cuda else y
    y_np = y_cpu.to(dtype=torch.float32).numpy()
    
    mse = np.mean((y_np - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_np - y_pred) ** 2)
    ss_tot = np.sum((y_np - np.mean(y_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return float(rmse), float(r2), y_pred


def evaluate_classification(model, X, y, temp_to_class, class_to_temp, X_mean, X_std, device):
    """Evaluate classification probe."""
    model.eval()
    
    # Move X to device for model inference if needed
    X_scaled = (X - X_mean.to(device=X.device, dtype=X.dtype)) / X_std.to(device=X.device, dtype=X.dtype)
    X_scaled = X_scaled.to(device, dtype=torch.float32, non_blocking=True)
    
    with torch.no_grad():
        logits = model(X_scaled).to(dtype=torch.float32)
        y_pred_classes = logits.argmax(dim=1).cpu().numpy()
    
    # Handle both CPU and CUDA tensors
    y_cpu = y.cpu() if y.is_cuda else y
    y_classes = np.array([temp_to_class[y_item.item()] for y_item in y_cpu])
    
    accuracy = accuracy_score(y_classes, y_pred_classes)
    conf_matrix = confusion_matrix(y_classes, y_pred_classes)
    
    return float(accuracy), conf_matrix, y_pred_classes


def main():
    parser = argparse.ArgumentParser(description='Train temperature probes (regression + classification)')
    parser.add_argument('--activations_dir', type=str, required=True,
                        help='Path to activations directory')
    parser.add_argument('--layers', type=str, required=True,
                        help='Layer indices to train (e.g., "19" or "0,1,2,3" or "0-3")')
    parser.add_argument('--max_files_per_temp', type=int, default=None,
                        help='Max files per temperature (default: all)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation set proportion of train+val (default: 0.2)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Max epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--output_dir', type=str, default='probe_results',
                        help='Base output directory (default: probe_results)')
    parser.add_argument('--preload_to_gpu', action='store_true',
                        help='Preload data to GPU after loading (reduces RAM usage, speeds up training)')
    parser.add_argument('--files_per_batch', type=int, default=4,
                        help='Number of activation files to process together before cleanup (default: 4)')
    parser.add_argument('--chunk_size', type=int, default=8,
                        help='Number of files to accumulate per layer before concatenating (default: 8)')
    parser.add_argument('--sample_tokens_per_sequence', type=int, default=50,
                        help='Evenly sample this many tokens per activation sequence (default: 50; set <=0 to keep all)')
    
    args = parser.parse_args()
    
    # Parse layer indices
    layer_indices = []
    for part in args.layers.split(','):
        part = part.strip()
        if '-' in part:
            # Range like "0-3"
            start, end = part.split('-')
            layer_indices.extend(range(int(start), int(end) + 1))
        else:
            # Single number
            layer_indices.append(int(part))
    
    layer_indices = sorted(list(set(layer_indices)))
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training layers: {layer_indices}")
    print(f"Files per batch: {args.files_per_batch}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Sample tokens per sequence: {args.sample_tokens_per_sequence if args.sample_tokens_per_sequence > 0 else 'All'}")
    
    # Create organized output directory structure
    # Extract model name from activations_dir path
    activations_path = Path(args.activations_dir)
    model_name = activations_path.name  # e.g., "Qwen3-0.6B"
    
    # Create descriptive subdirectory name
    max_files_str = f"{args.max_files_per_temp}files" if args.max_files_per_temp else "allfiles"
    sample_str = f"{args.sample_tokens_per_sequence}tokens" if args.sample_tokens_per_sequence and args.sample_tokens_per_sequence > 0 else "alltokens"
    experiment_name = f"{model_name}_{max_files_str}_{sample_str}"
    
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Save experiment config
    config_path = output_dir / 'experiment_config.json'
    experiment_config = {
        'model_name': model_name,
        'activations_dir': str(args.activations_dir),
        'max_files_per_temp': args.max_files_per_temp,
        'sample_tokens_per_sequence': args.sample_tokens_per_sequence,
        'test_size': args.test_size,
        'val_size': args.val_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'patience': args.patience,
        'device': args.device,
        'layers_trained': layer_indices,
        'files_per_batch': args.files_per_batch,
        'chunk_size': args.chunk_size,
    }
    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    print(f"Saved experiment config to {config_path}\n")
    
    # Load data (file-wise split to avoid leakage)
    # Load all requested layers at once (efficient - reads each file only once)
    print(f"\n{'='*80}")
    print(f"Loading data for layers {layer_indices}")
    print(f"{'='*80}\n")
    
    # Determine preload device
    preload_device = 'cuda' if args.preload_to_gpu and torch.cuda.is_available() else 'cpu'
    
    data_by_layer, temp_to_class, class_to_temp = load_data_by_files(
        args.activations_dir,
        layer_indices,
        args.sample_tokens_per_sequence,
        args.max_files_per_temp,
        args.test_size,
        args.val_size,
        preload_device=preload_device,
        files_per_batch=args.files_per_batch,
        chunk_size=args.chunk_size,
    )
    
    # Debug: class distribution before training
    if data_by_layer:
        sample_layer = next(iter(data_by_layer.keys()), None)
        if sample_layer is not None and sample_layer in data_by_layer:
            print("\nClass distribution before training:")
            distributions = {
                'train': data_by_layer[sample_layer]['train'][1],
                'val': data_by_layer[sample_layer]['val'][1],
                'test': data_by_layer[sample_layer]['test'][1],
            }
            for split_name, y_tensor in distributions.items():
                if y_tensor is None:
                    print(f"  {split_name}: no data")
                    continue
                y_cpu = y_tensor.detach().cpu()
                total = int(y_cpu.numel())
                print(f"  {split_name} total: {total}")
                for temp in sorted(temp_to_class.keys()):
                    count = int((y_cpu == temp).sum().item())
                    print(f"    Temp {temp}: {count}")

    # Train probes for each layer
    for layer_idx in layer_indices:
        if layer_idx not in data_by_layer:
            print(f"Warning: No data for layer {layer_idx}, skipping")
            continue
        
        print(f"\n{'='*80}")
        print(f"Layer {layer_idx}")
        print(f"{'='*80}\n")
        
        X_train, y_train = data_by_layer[layer_idx]['train']
        X_val, y_val = data_by_layer[layer_idx]['val']
        X_test, y_test = data_by_layer[layer_idx]['test']
        
        # Train regression probe
        print("Training Regression Probe...")
        reg_model, reg_mean, reg_std = train_regression_probe(
            X_train, y_train, X_val, y_val, device, 
            args.epochs, args.lr, args.patience
        )
        
        # Evaluate regression
        train_rmse, train_r2, _ = evaluate_regression(reg_model, X_train, y_train, reg_mean, reg_std, device)
        test_rmse, test_r2, test_preds_reg = evaluate_regression(reg_model, X_test, y_test, reg_mean, reg_std, device)
        
        print(f"\nRegression Results:")
        print(f"  Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"  Test RMSE:  {test_rmse:.4f}, R²: {test_r2:.4f}")
        
        # Train classification probe
        print("\nTraining Classification Probe...")
        clf_model, clf_mean, clf_std = train_classification_probe(
            X_train, y_train, X_val, y_val, temp_to_class, class_to_temp, device,
            args.epochs, args.lr, args.patience
        )
        
        # Evaluate classification
        train_acc, _, _ = evaluate_classification(clf_model, X_train, y_train, temp_to_class, class_to_temp, clf_mean, clf_std, device)
        test_acc, conf_matrix, test_preds_clf = evaluate_classification(clf_model, X_test, y_test, temp_to_class, class_to_temp, clf_mean, clf_std, device)
        
        print(f"\nClassification Results:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        
        # Save models
        reg_model_path = output_dir / 'models' / f'regression_layer_{layer_idx}.pt'
        torch.save({
            'model_state_dict': reg_model.state_dict(),
            'X_mean': reg_mean.detach().cpu().to(torch.float32),
            'X_std': reg_std.detach().cpu().to(torch.float32),
            'layer': layer_idx,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
        }, reg_model_path)
        print(f"\nSaved regression model to {reg_model_path}")
        
        clf_model_path = output_dir / 'models' / f'classification_layer_{layer_idx}.pt'
        torch.save({
            'model_state_dict': clf_model.state_dict(),
            'X_mean': clf_mean.detach().cpu().to(torch.float32),
            'X_std': clf_std.detach().cpu().to(torch.float32),
            'layer': layer_idx,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'temp_to_class': temp_to_class,
            'class_to_temp': class_to_temp,
        }, clf_model_path)
        print(f"Saved classification model to {clf_model_path}")
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Get y_test as numpy (handle GPU tensors)
        y_test_np = y_test.cpu().numpy() if y_test.is_cuda else y_test.numpy()
        
        # Regression scatter plot
        ax1 = axes[0]
        ax1.scatter(y_test_np, test_preds_reg, alpha=0.3, s=10)
        ax1.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', label='Perfect prediction')
        ax1.set_xlabel('Actual Temperature')
        ax1.set_ylabel('Predicted Temperature')
        ax1.set_title(f'Regression Probe - Layer {layer_idx}\nRMSE={test_rmse:.4f}, R²={test_r2:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Classification confusion matrix
        ax2 = axes[1]
        temp_labels = [f'{class_to_temp[i]:.2f}' for i in range(len(class_to_temp))]
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=temp_labels, yticklabels=temp_labels)
        ax2.set_xlabel('Predicted Temperature')
        ax2.set_ylabel('Actual Temperature')
        ax2.set_title(f'Classification Probe - Layer {layer_idx}\nAccuracy={test_acc:.4f}')
        
        plt.tight_layout()
        plot_path = output_dir / 'plots' / f'layer_{layer_idx}_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plots to {plot_path}")
        plt.close()
        
        # Save results JSON
        results = {
            'layer': layer_idx,
            'regression': {
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
            },
            'classification': {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'confusion_matrix': conf_matrix.tolist(),
            },
            'config': vars(args),
        }
        
        results_path = output_dir / f'layer_{layer_idx}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_path}")
        
        # Clean up to save memory aggressively
        del reg_model, clf_model, X_train, y_train, X_val, y_val, X_test, y_test
        del reg_mean, reg_std, clf_mean, clf_std
        del train_rmse, train_r2, test_rmse, test_r2
        del train_acc, test_acc, conf_matrix, test_preds_reg, test_preds_clf
        aggressive_gc()
        
        print(f"Memory cleaned up after layer {layer_idx}")
    
    # Clean up all loaded data
    del data_by_layer
    aggressive_gc()
    
    print("\nFinal memory cleanup...")
    aggressive_gc()
    
    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

