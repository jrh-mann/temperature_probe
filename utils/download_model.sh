#!/bin/bash

# Download gpt-oss-20b model from HuggingFace
# This script downloads the model to a local directory for use with vLLM

# Activate virtual environment
source /root/counterfactual_steering/.venv/bin/activate

# Default configuration
MODEL_NAME="${1:-${MODEL_NAME:-openai/gpt-oss-20b}}"
LOCAL_DIR="${LOCAL_DIR:-/workspace}"
CACHE_DIR="${CACHE_DIR:-/workspace/.cache/huggingface}"

echo "Downloading model: $MODEL_NAME"
echo "Local directory: $LOCAL_DIR"
echo "Cache directory: $CACHE_DIR"
echo ""

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install -U "huggingface_hub[cli]"
fi

# Create local directory and cache directory if they don't exist
mkdir -p "$LOCAL_DIR"
mkdir -p "$CACHE_DIR"

# Set HuggingFace cache directory to use workspace (prevents disk space issues)
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"

# Check available disk space
echo "Checking disk space..."
df -h "$(dirname "$LOCAL_DIR")" | tail -1
echo ""

# Download the model using Python API (more reliable than CLI)
echo "Starting download (this may take a while)..."
# Note: HF_HOME and HF_HUB_CACHE are set above to ensure cache goes to /workspace
python << EOF
import os
import sys
from huggingface_hub import snapshot_download

model_name = "$MODEL_NAME"
local_dir = "$LOCAL_DIR"
cache_dir = "$CACHE_DIR"

# Set environment variables for cache location
os.environ["HF_HOME"] = cache_dir
os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")

print(f"Downloading {model_name} to {local_dir}...")
try:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        cache_dir=cache_dir
    )
    print(f"\n✓ Model downloaded successfully to {local_dir}")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    sys.exit(1)
EOF

DOWNLOAD_STATUS=$?

if [ $DOWNLOAD_STATUS -eq 0 ]; then
    echo ""
    echo "Local path: $LOCAL_DIR"
    echo ""
    echo "To use this model, set MODEL_PATH in start_vllm_server.sh:"
    echo "  export MODEL_PATH=\"$LOCAL_DIR\""
    echo "  ./start_vllm_server.sh"
else
    echo ""
    echo "✗ Download failed. Please check:"
    echo "  1. You have internet connection"
    echo "  2. You have enough disk space (model is ~40GB)"
    echo "  3. You have access to the model on HuggingFace"
    exit 1
fi