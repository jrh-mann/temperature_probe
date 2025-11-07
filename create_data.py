"""Generate rollouts from training data at different temperatures."""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from utils.vllm_client import VLLMClient
from utils.activations import store_activations


def generate_rollouts_from_harmless(
    model_name: str = "Qwen3-0.6B",
    temperatures: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0],
    num_samples: int = 100,
    data_file: str = "data/harmless_train.json",
    rollout_dir: str = "rollouts",
    vllm_base_url: str = "http://localhost:8000/v1",
    max_tokens: int = 512,
    max_workers: int = 10,
    **generation_kwargs
) -> None:
    """
    Generate rollouts from harmless_train.json at different temperatures.
    
    Args:
        model_name: Name of the model (used for directory structure)
        temperatures: List of temperatures to use for generation
        num_samples: Number of samples to take from the dataset (first N)
        data_file: Path to the harmless_train.json file
        rollout_dir: Base directory for saving rollouts
        vllm_base_url: Base URL for the vLLM server
        max_tokens: Maximum tokens to generate per response
        max_workers: Maximum number of parallel workers for generation
        **generation_kwargs: Additional arguments to pass to the generation function
    """
    # Read the data file
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Reading data from {data_file}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Take first num_samples
    samples = all_data[:num_samples]
    print(f"Using first {len(samples)} samples from dataset")
    
    # Extract instructions
    instructions = [sample.get("instruction", "") for sample in samples]
    if not all(instructions):
        print("Warning: Some samples are missing 'instruction' field")
    
    # Initialize vLLM client
    print(f"Connecting to vLLM server at {vllm_base_url}...")
    client = VLLMClient(base_url=vllm_base_url)
    
    # Get available models from the server
    available_models = client.get_available_models()
    if not available_models:
        raise RuntimeError("No models available on the vLLM server. Please check the server is running and has models loaded.")
    
    print(f"Available models on server: {available_models}")
    
    # Note: When vLLM loads a model from a local path (MODEL_PATH), 
    # the model ID will be that path (e.g., '/workspace') rather than a model name.
    # This is expected behavior.
    
    # Determine which model to use for generation
    # If model_name is in available models, use it; otherwise use the first available model
    if model_name in available_models:
        actual_model = model_name
        print(f"Using specified model: {actual_model}")
    else:
        # Use the first available model from the server
        actual_model = available_models[0]
        print(f"Model '{model_name}' not found on server.")
        if actual_model.startswith('/'):
            print(f"  Server is using a local path-based model ID: {actual_model}")
            print(f"  (This happens when vLLM was started with MODEL_PATH rather than MODEL_NAME)")
        print(f"  Using server model: {actual_model}")
        print(f"  Note: Rollouts will still be saved under '{model_name}' directory structure")
    
    # Create rollout directory structure
    base_rollout_dir = Path(rollout_dir)
    model_dir = base_rollout_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate rollouts for each temperature
    for temp in temperatures:
        print(f"\nGenerating rollouts at temperature {temp}...")
        temp_dir = model_dir / f"temperature_{temp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate responses
        print(f"Generating responses for {len(instructions)} prompts...")
        responses = client.generate(
            prompts=instructions,
            model=actual_model,
            temperature=temp,
            max_tokens=max_tokens,
            max_workers=max_workers,
            return_full_response=True,
            **generation_kwargs
        )
        
        # Prepare rollout data
        rollout_data = []
        for i, (sample, response) in enumerate(zip(samples, responses)):
            if isinstance(response, dict):
                rollout_item = {
                    "sample_index": i,
                    "instruction": sample.get("instruction", ""),
                    "category": sample.get("category"),
                    "temperature": temp,
                    "model": response.get("model", actual_model),
                    "completion": response.get("completion", ""),
                    "full_text": response.get("full_text", response.get("completion", "")),
                    "finish_reason": response.get("finish_reason"),
                    "usage": response.get("usage", {}),
                }
                
                # Add optional fields if present
                if "prefill" in response and response["prefill"]:
                    rollout_item["prefill"] = response["prefill"]
                if "reasoning" in response and response["reasoning"]:
                    rollout_item["reasoning"] = response["reasoning"]
                if "logprobs" in response:
                    rollout_item["logprobs"] = response["logprobs"]
            else:
                # Handle error case
                rollout_item = {
                    "sample_index": i,
                    "instruction": sample.get("instruction", ""),
                    "category": sample.get("category"),
                    "temperature": temp,
                    "model": actual_model,
                    "completion": str(response),
                    "error": True
                }
            
            rollout_data.append(rollout_item)
        
        # Save rollout data
        rollout_file = temp_dir / "rollouts.json"
        with open(rollout_file, 'w', encoding='utf-8') as f:
            json.dump(rollout_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(rollout_data)} rollouts to {rollout_file}")
    
    print("\nAll rollouts generated successfully!")


def extract_activations_from_rollouts(
    rollout_dir: str = "rollouts",
    activations_dir: str = "/workspace/activations",
    model_path: str = "/workspace",
    device: str = "cuda",
    n_samples: int = None,
    layer_indices: List[int] = None,
) -> None:
    """
    Extract activations from all rollouts and save them organized by model and temperature.
    
    This function:
    1. Scans the rollouts directory for all model/temperature combinations
    2. Loads the model using nnsight
    3. Extracts prompts from rollouts
    4. Formats prompts using the tokenizer's chat template
    5. Extracts activations for all prompts
    6. Saves activations to activations_dir/{model_name}/temperature_{temp}/
    
    Args:
        rollout_dir: Base directory containing rollouts (organized as {model_name}/temperature_{temp}/rollouts.json)
        activations_dir: Base directory for saving activations
        model_path: Path to the model directory (used for loading with nnsight)
        device: Device to run the model on ('cuda' or 'cpu')
    """
    try:
        from nnsight import LanguageModel
    except ImportError:
        raise ImportError("nnsight is required. Install it with: pip install nnsight")
    
    rollout_base = Path(rollout_dir)
    activations_base = Path(activations_dir)
    activations_base.mkdir(parents=True, exist_ok=True)
    
    if not rollout_base.exists():
        raise FileNotFoundError(f"Rollouts directory not found: {rollout_dir}")
    
    # Check if the provided path is already a model directory (contains temperature_* directories)
    # or if it's the base directory (contains model name directories)
    temp_dirs_in_base = [d for d in rollout_base.iterdir() if d.is_dir() and d.name.startswith("temperature_")]
    
    if temp_dirs_in_base:
        # The provided path is already a model directory
        model_name = rollout_base.name
        model_dir = rollout_base
        model_dirs = [model_dir]
        print(f"Detected model directory: {model_name}")
    else:
        # The provided path is the base directory, find all model directories
        model_dirs = [d for d in rollout_base.iterdir() if d.is_dir()]
        if not model_dirs:
            print(f"No model directories found in {rollout_dir}")
            return
        print(f"Found {len(model_dirs)} model directories")
    
    # Load the model once (assuming all temperatures use the same model)
    print(f"\nLoading model from {model_path}...")
    try:
        model = LanguageModel(model_path, device_map=device)
        print(f"Model loaded successfully. Tokenizer: {model.tokenizer.__class__.__name__}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    # Process each model directory
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Find all temperature directories
        temp_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("temperature_")]
        if not temp_dirs:
            print(f"  No temperature directories found in {model_dir}")
            continue
        
        print(f"  Found {len(temp_dirs)} temperature directories")
        
        # Process each temperature directory
        for temp_dir in sorted(temp_dirs):
            rollout_file = temp_dir / "rollouts.json"
            if not rollout_file.exists():
                print(f"  Warning: {rollout_file} not found, skipping")
                continue
            
            # Extract temperature from directory name
            try:
                temp = float(temp_dir.name.replace("temperature_", ""))
            except ValueError:
                print(f"  Warning: Could not parse temperature from {temp_dir.name}, skipping")
                continue
            
            print(f"\n  Processing temperature {temp}...")
            
            # Load rollouts
            print(f"    Loading rollouts from {rollout_file}...")
            with open(rollout_file, 'r', encoding='utf-8') as f:
                rollouts = json.load(f)
            
            if not rollouts:
                print(f"    No rollouts found in {rollout_file}, skipping")
                continue
            
            print(f"    Found {len(rollouts)} rollouts")

            # Randomly sample 100 rollouts
            if n_samples:
                rollouts = rollouts[:n_samples]
                print(f"    Reduced to first {n_samples}")
            
            # Extract instructions and format prompts
            print(f"    Formatting prompts...")
            prompts = []
            for rollout in rollouts:
                instruction = rollout.get("instruction", "")
                response = rollout.get("completion", "")
                if not instruction:
                    print(f"    Warning: Empty instruction found in rollout, skipping")
                    continue
                
                # Format prompt using chat template
                try:
                    formatted_prompt = model.tokenizer.apply_chat_template(
                        [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}],
                        tokenize=False,
                    )
                    prompts.append(formatted_prompt)
                except Exception as e:
                    print(f"    Warning: Failed to format prompt: {e}, using raw instruction")
                    prompts.append(instruction)
            
            if not prompts:
                print(f"    No valid prompts found, skipping")
                continue
            
            print(f"    Formatted {len(prompts)} prompts")
            
            # Create output directory
            output_dir = activations_base / model_name / temp_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract activations
            print(f"    Extracting activations...")
            print(f"    Saving to {output_dir}")
            try:
                store_activations(model, prompts, str(output_dir))
                # Verify files were created
                saved_files = list(output_dir.glob("*.pt"))
                if saved_files:
                    print(f"    ✓ Successfully extracted activations for temperature {temp}")
                    print(f"    ✓ Saved {len(saved_files)} activation files")
                else:
                    print(f"    ⚠ Warning: No activation files were created in {output_dir}")
            except Exception as e:
                print(f"    ✗ Error extracting activations: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*60}")
    print("Finished processing all rollouts!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Example usage
    generate_rollouts_from_harmless(
        model_name="Qwen3-0.6B",
        temperatures=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        num_samples=1000,
        max_tokens=10000
    )
    extract_activations_from_rollouts(
        rollout_dir="rollouts/Qwen3-0.6B",
        activations_dir="/workspace/activations",
        model_path="/workspace",
        device="cuda",
        layer_indices=[i for i in range(25, 33)],
    )