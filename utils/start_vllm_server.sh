#!/bin/bash

# Start vLLM server with OpenAI-compatible API
# This script starts an OpenAI-compatible API server
#
# To use a different model, set the MODEL_NAME environment variable:
#   export MODEL_NAME="microsoft/phi-2"  # or any other HuggingFace model
#   ./start_vllm_server.sh
#
# Common models:
#   - microsoft/phi-2
#   - microsoft/phi-1_5
#   - TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   - gpt2  (if using local path: /path/to/model)

# Activate virtual environment
source /root/temperature_probe/.venv/bin/activate

# Default configuration
# IMPORTANT: Set MODEL_NAME or MODEL_PATH
#   - MODEL_NAME: HuggingFace model identifier (e.g., "openai/gpt-oss-20b")
#   - MODEL_PATH: Local path to downloaded model (e.g., "/workspace" or "/workspace/model-name")
#   MODEL_PATH takes precedence if both are set

# Default to a small, fast model that's publicly available
# For larger models, download first using download_model.sh
MODEL_NAME="${MODEL_NAME:-microsoft/phi-2}"
MODEL_PATH="${MODEL_PATH:-}"

# If MODEL_PATH is not set but MODEL_NAME looks like it might be in /workspace, check
if [ -z "$MODEL_PATH" ] && [[ "$MODEL_NAME" == *"/"* ]]; then
    # Extract model folder name
    MODEL_FOLDER_NAME=$(echo "$MODEL_NAME" | sed 's|.*/||' | sed 's|[^a-zA-Z0-9._-]|_|g')
    POTENTIAL_PATH="/workspace/${MODEL_FOLDER_NAME}"
    
    if [ -d "$POTENTIAL_PATH" ]; then
        echo "Found local model at $POTENTIAL_PATH"
        MODEL_PATH="$POTENTIAL_PATH"
    fi
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"

# Use MODEL_PATH if set, otherwise use MODEL_NAME
MODEL="${MODEL_PATH:-$MODEL_NAME}"

echo "Starting vLLM server with configuration"
echo "  Model: $MODEL"
if [ -n "$MODEL_PATH" ]; then
    echo "  (Using local path: $MODEL_PATH)"
else
    echo "  (Using HuggingFace identifier: $MODEL_NAME)"
fi
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE GPUs"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo ""
if [ -z "$MODEL_PATH" ]; then
    echo "Note: To use a local model, first download it:"
    echo "      bash utils/download_model.sh <model-name>"
    echo "      Then: MODEL_PATH='/workspace/<model-folder>' bash utils/start_vllm_server.sh"
    echo ""
    echo "Example:"
    echo "      bash utils/download_model.sh openai/gpt-oss-20b"
    echo "      MODEL_PATH='/workspace/gpt-oss-20b' bash utils/start_vllm_server.sh"
    echo ""
fi

# Start the server
vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"