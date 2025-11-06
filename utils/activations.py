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

def store_activations(model, prompts, output_dir):
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
                    for layer in model.model.layers:
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

def load_activations(output_dir, max_workers=8):
    acts = [act for act in os.listdir(output_dir) if act != "prompts.pt"]
    
    def load_single_file(filename):
        return torch.load(os.path.join(output_dir, filename))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use list() to force execution and maintain order
        results = list(tqdm(
            executor.map(load_single_file, acts),
            total=len(acts),
            desc="Loading activations"
        ))
    
    return results