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
source /root/counterfactual_steering/.venv/bin/activate

# Default configuration
# IMPORTANT: Set MODEL_NAME or MODEL_PATH
#   - MODEL_NAME: HuggingFace model identifier (e.g., "openai/gpt-oss-20b")
#   - MODEL_PATH: Local path to downloaded model (e.g., "/workspace" or "/workspace/model-name")
#   MODEL_PATH takes precedence if both are set

MODEL_PATH="${MODEL_PATH:-}"
# Default to a small, fast model that's publicly available
# For gpt-oss-20b, you'll need to download it first using download_model.sh
MODEL_NAME="${MODEL_NAME:-microsoft/phi-2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"

# Use MODEL_PATH if set, otherwise use MODEL_NAME
MODEL="${MODEL_PATH:-$MODEL_NAME}"

echo "Starting vLLM server with configuration:"
echo "  Model: $MODEL"
if [ -n "$MODEL_PATH" ]; then
    echo "  (Using local path)"
else
    echo "  (Using HuggingFace identifier)"
fi
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE GPUs"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo ""
if [ -z "$MODEL_PATH" ]; then
    echo "Note: To use a local model, first download it:"
    echo "      ./download_model.sh <model-name>"
    echo "      Then: export MODEL_PATH='/workspace' && ./start_vllm_server.sh"
    echo "      (or set MODEL_PATH to the exact path where the model was downloaded)"
    echo ""
fi

# Start the server
vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"