from tqdm import tqdm
import gc
import os
import torch
from concurrent.futures import ThreadPoolExecutor

def apply_chat_template(tokenizer, prompt, reasoning, completion):
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True
    ) + "<|channel|>analysis<|message|>" + \
    reasoning + "<|end|><|start|>assistant<|channel|>final<|message|>" + \
    completion + "<|return|>"

def get_start_of_sublist(tokenizer, prompt):
    tokens = tokenizer.tokenize(prompt)
    target = ['<|channel|>', 'analysis', '<|message|>']
    for i in range(len(tokens) - len(target) + 1):
        if tokens[i:i+len(target)] == target:
            return i + 3
    raise ValueError("Not found")

def store_activations(model, prompts, output_dir, layer_indices=None):
    # Convert to string if Path object
    output_dir = str(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(prompts, os.path.join(output_dir, "prompts.pt"))

    with torch.no_grad():
        for index, prompt in enumerate(tqdm(prompts, desc="Extracting activations")):
            residual_stream = []
            #start_of_response = get_start_of_sublist(model.tokenizer, prompt)
            try:
                with model.trace(prompt) as tracer:
                    # Save activations from each layer

                    if layer_indices is None:
                        layers = model.model.layers
                    else:
                        layers = [model.model.layers[i] for i in layer_indices]
                    for layer in layers:
                        residual_stream.append(layer.output.save())
                    # Execute the trace by accessing the output
                
                # After trace execution, extract the saved values
                # In nnsight, saved proxies are populated after execution
                # The saved object should be the tensor itself after execution
                # Original code used [0] indexing, so saved might return a tuple/list
                acts_list = []
                for saved in residual_stream:
                    # Try different ways to access the saved tensor
                    tensor = None
                    try:
                        # Try direct access first (most common case)
                        if isinstance(saved, torch.Tensor):
                            tensor = saved
                        # Try [0] indexing (as in original code)
                        elif hasattr(saved, '__getitem__'):
                            try:
                                tensor = saved[0]
                            except (IndexError, TypeError):
                                pass
                        # Try .value attribute
                        if tensor is None and hasattr(saved, 'value') and saved.value is not None:
                            tensor = saved.value
                        # Try .output attribute
                        if tensor is None and hasattr(saved, 'output') and saved.output is not None:
                            tensor = saved.output
                        # If still None, try direct access
                        if tensor is None:
                            tensor = saved
                        
                        if tensor is not None and isinstance(tensor, torch.Tensor):
                            acts_list.append(tensor)
                        else:
                            print(f"Warning: Could not extract tensor from saved value (type: {type(saved)})")
                    except Exception as e:
                        print(f"Warning: Could not extract saved value: {e}, trying direct access")
                        # Last resort: try direct access
                        if isinstance(saved, torch.Tensor):
                            acts_list.append(saved)
                
                if not acts_list:
                    raise ValueError("No activations were extracted from any layer")
                
                acts = torch.stack(acts_list)
                #indexed_acts = acts[:,start_of_response:]
                cpuacts = acts.cpu()
                save_path = os.path.join(output_dir, f"{index}.pt")
                torch.save(cpuacts, save_path)

                del residual_stream
                del acts_list
                del acts
                del cpuacts
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"\nError processing prompt {index}: {e}")
                import traceback
                traceback.print_exc()
                raise

def load_activations(output_dir, layer_idx=None, layer_indices=None, token_every_n=1, max_workers=4, max_files=None):
    """
    Load activations from a directory, optionally for specific layers.
    Memory-safe: processes files sequentially to avoid OOM.
    
    Args:
        output_dir: Directory containing .pt activation files
        layer_idx: If specified (int), return only that layer. (For backward compatibility)
        layer_indices: If specified (list), return only those layers. Takes precedence over layer_idx.
                       If both are None, return all layers.
        token_every_n: Sample every nth token (1 = all tokens, 2 = every 2nd, etc.)
        max_workers: Number of parallel workers (default 4)
        max_files: If specified, only load the first N files (for faster testing)
    
    Returns:
        List of tensors:
        - If layer_idx specified: each tensor is (seq_len_sampled, hidden_dim)
        - If layer_indices specified: each tensor is (len(layer_indices), seq_len_sampled, hidden_dim)
        - If neither specified: each tensor is (num_layers, seq_len_sampled, hidden_dim)
    """
    act_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pt') and f != "prompts.pt"])
    
    # Limit number of files if requested
    if max_files is not None and max_files > 0:
        act_files = act_files[:max_files]
    
    if not act_files:
        raise ValueError(f"No activation files found in {output_dir}")
    
    # Determine which layers to load
    if layer_indices is not None:
        layers_to_load = sorted(set(layer_indices))
        load_mode = 'multiple'
    elif layer_idx is not None:
        layers_to_load = [layer_idx]
        load_mode = 'single'
    else:
        layers_to_load = None
        load_mode = 'all'
    
    def load_single_file(filename):
        acts = torch.load(os.path.join(output_dir, filename), map_location='cpu')
        # acts shape: (num_layers, batch=1, seq_len, hidden_dim)
        
        try:
            if load_mode == 'single':
                # Extract single layer: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
                if layers_to_load[0] >= acts.shape[0]:
                    return torch.empty((0, acts.shape[3]), dtype=acts.dtype)
                layer_acts = acts[layers_to_load[0]].squeeze(0)  # (seq_len, hidden_dim)
                
            elif load_mode == 'multiple':
                # Extract only requested layers: stack into (num_extracted, seq_len, hidden_dim)
                # Track which layers were actually extracted (some might not exist in file)
                extracted_layers = []
                extracted_indices = []  # Track which layer indices were extracted
                for idx in layers_to_load:
                    if idx < acts.shape[0]:
                        layer = acts[idx].squeeze(0)  # (seq_len, hidden_dim)
                        extracted_layers.append(layer)
                        extracted_indices.append(idx)
                
                if not extracted_layers:
                    return torch.empty((0, acts.shape[3]), dtype=acts.dtype), []
                
                # Stack: (num_extracted, seq_len, hidden_dim)
                layer_acts = torch.stack(extracted_layers, dim=0)
                
                # Uniform sampling: take every nth token
                if token_every_n > 1:
                    layer_acts = layer_acts[:, ::token_every_n]  # Apply to seq_len dimension
                
                # Return both the tensor and the mapping of positions to layer indices
                return layer_acts, extracted_indices
                
            else:  # load_mode == 'all'
                # Keep all layers: (num_layers, seq_len, hidden_dim)
                layer_acts = acts.squeeze(1)  # Remove batch dimension
            
            # Uniform sampling: take every nth token (for single and 'all' modes)
            if token_every_n > 1:
                if load_mode == 'single':
                    layer_acts = layer_acts[::token_every_n]
                else:  # 'all' mode
                    layer_acts = layer_acts[:, ::token_every_n]  # Apply to seq_len dimension
            
            # For single and 'all' modes, return just the tensor
            return layer_acts
            
        finally:
            # Explicit cleanup
            del acts
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    # Use parallel loading for speed (default 4 workers, can be increased)
    desc_str = f"Loading activations"
    if load_mode == 'single':
        desc_str += f" (layer={layers_to_load[0]}, every_n={token_every_n})"
    elif load_mode == 'multiple':
        desc_str += f" (layers={layers_to_load}, every_n={token_every_n})"
    else:
        desc_str += f" (all layers, every_n={token_every_n})"
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(load_single_file, act_files),
            total=len(act_files),
            desc=desc_str
        ))
    
    # For 'multiple' mode, results is a list of (tensor, extracted_indices) tuples
    # For other modes, results is a list of tensors
    if load_mode == 'multiple':
        # Unpack: separate tensors and indices lists
        result_tensors = [r[0] for r in results]
        # All files should have the same extracted_indices (same layers exist in all files)
        # But we'll handle per-file variation in train_probe.py
        return result_tensors, results[0][1] if results else []
    else:
        return results