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

def store_activations(
    model,
    prompts,
    output_dir,
    layer_indices=None,
    batch_size=4,
    save_workers=4,
    save_dtype=torch.bfloat16,
):
    """Trace `prompts` through `model` and persist per-prompt layer activations.

    Args:
        model: nnsight LanguageModel instance.
        prompts: Iterable of formatted prompt strings.
        output_dir: Directory path to write activation tensors and metadata.
        layer_indices: Optional iterable of layer indices to capture. Defaults to all layers.
        batch_size: Number of prompts to run per forward pass.
        save_workers: Number of background threads used to write tensors to disk.
        save_dtype: If provided, tensors are cast to this dtype before saving (default: bfloat16).
    """
    # Convert to string if Path object
    output_dir = str(output_dir)

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if save_workers < 1:
        raise ValueError("save_workers must be at least 1")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(prompts, os.path.join(output_dir, "prompts.pt"))

    def _extract_tensor(saved_value):
        tensor = None
        try:
            if isinstance(saved_value, torch.Tensor):
                tensor = saved_value
            elif hasattr(saved_value, '__getitem__'):
                try:
                    tensor = saved_value[0]
                except (IndexError, TypeError):
                    tensor = None
            if tensor is None and hasattr(saved_value, 'value') and saved_value.value is not None:
                tensor = saved_value.value
            if tensor is None and hasattr(saved_value, 'output') and saved_value.output is not None:
                tensor = saved_value.output
            if tensor is None and isinstance(saved_value, torch.Tensor):
                tensor = saved_value
        except Exception as e:
            print(f"Warning: Could not extract saved value: {e}, trying direct access")
            if isinstance(saved_value, torch.Tensor):
                tensor = saved_value
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            print(f"Warning: Could not extract tensor from saved value (type: {type(saved_value)})")
            tensor = None
        return tensor

    save_executor = ThreadPoolExecutor(max_workers=save_workers)
    pending_saves = []

    def _flush_pending(force=False):
        if not pending_saves:
            return
        if force:
            while pending_saves:
                future = pending_saves.pop(0)
                future.result()
        elif len(pending_saves) >= save_workers * 2:
            future = pending_saves.pop(0)
            future.result()

    prompts = list(prompts)

    try:
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Extracting activations"):
                batch_prompts = prompts[batch_start: batch_start + batch_size]
                residual_stream = []

                try:
                    trace_input = batch_prompts if len(batch_prompts) > 1 else batch_prompts[0]
                    with model.trace(trace_input) as tracer:
                        if layer_indices is None:
                            layers = model.model.layers
                        else:
                            layers = [model.model.layers[i] for i in layer_indices]
                        for layer in layers:
                            residual_stream.append(layer.output.save())

                    acts_list = []
                    for saved in residual_stream:
                        tensor = _extract_tensor(saved)
                        if tensor is not None:
                            acts_list.append(tensor)

                    if not acts_list:
                        raise ValueError("No activations were extracted from any layer")

                    acts = torch.stack(acts_list)
                    cpuacts = acts.cpu()

                    if cpuacts.ndim == 4:
                        batch_dim = cpuacts.shape[1]

                        def _per_prompt_generator():
                            for idx in range(batch_dim):
                                yield idx, cpuacts[:, idx].contiguous().clone()

                        per_prompt_iter = _per_prompt_generator()
                    elif cpuacts.ndim == 3:
                        batch_dim = 1

                        def _single_prompt_generator():
                            yield 0, cpuacts.contiguous().clone()

                        per_prompt_iter = _single_prompt_generator()
                    else:
                        raise ValueError(f"Unexpected activation shape: {cpuacts.shape}")

                    if batch_dim != len(batch_prompts):
                        print(
                            f"Warning: Batch size mismatch (expected {len(batch_prompts)}, got {batch_dim})."
                        )

                    for idx_within_batch, tensor_to_save in per_prompt_iter:
                        if save_dtype is not None and tensor_to_save.dtype != save_dtype:
                            tensor_to_save = tensor_to_save.to(dtype=save_dtype)
                        prompt_index = batch_start + idx_within_batch
                        save_path = os.path.join(output_dir, f"{prompt_index}.pt")
                        future = save_executor.submit(torch.save, tensor_to_save, save_path)
                        pending_saves.append(future)
                        _flush_pending()
                        del tensor_to_save

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"\nError processing prompts {batch_start}:{batch_start + len(batch_prompts)}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                finally:
                    residual_stream.clear()
                    acts_list = []
                    if 'acts' in locals():
                        del acts
                    if 'cpuacts' in locals():
                        del cpuacts
                    if 'per_prompt_iter' in locals():
                        del per_prompt_iter
                    if 'trace_input' in locals():
                        del trace_input
                    _flush_pending()
                    gc.collect()
    finally:
        _flush_pending(force=True)
        save_executor.shutdown(wait=True)

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