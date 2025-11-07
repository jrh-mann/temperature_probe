#!/bin/bash
# Train probes for all layers with proper file-wise splitting

ACTIVATIONS_DIR="${1:-/workspace/activations/Qwen3-0.6B}"
MAX_FILES="${2:-30}"  # Limit files per temperature for faster training
TOKEN_EVERY_N="${3:-10}"

echo "Training probes for all layers"
echo "Activations dir: $ACTIVATIONS_DIR"
echo "Max files per temp: $MAX_FILES"
echo "Token sampling: every $TOKEN_EVERY_N"
echo ""

source /root/temperature_probe/.venv/bin/activate

# Detect number of layers by loading a sample file
FIRST_TEMP_DIR=$(ls -d "$ACTIVATIONS_DIR"/temperature_* | head -1)
FIRST_FILE=$(ls "$FIRST_TEMP_DIR"/*.pt | grep -v prompts | head -1)

# Use Python to detect number of layers
NUM_LAYERS=$(python3 -c "import torch; acts = torch.load('$FIRST_FILE', map_location='cpu'); print(acts.shape[0])")

echo "Detected $NUM_LAYERS layers"
echo ""

# Train layers in batches for efficiency (loads multiple layers at once)
BATCH_SIZE=7

for start_layer in $(seq 0 $BATCH_SIZE $((NUM_LAYERS - 1))); do
    end_layer=$((start_layer + BATCH_SIZE - 1))
    if [ $end_layer -ge $NUM_LAYERS ]; then
        end_layer=$((NUM_LAYERS - 1))
    fi
    
    echo "================================"
    echo "Training Layers $start_layer-$end_layer / $((NUM_LAYERS - 1))"
    echo "================================"
    
    python3 train_probe_v2.py \
        --activations_dir "$ACTIVATIONS_DIR" \
        --layers "$start_layer-$end_layer" \
        --token_every_n "$TOKEN_EVERY_N" \
        --max_files_per_temp "$MAX_FILES" \
        --device cuda \
        --preload_to_gpu \
        --epochs 1000 \
        --patience 50 \
        --output_dir "probe_results"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to train layers $start_layer-$end_layer"
        exit 1
    fi
    
    echo ""
done

echo "================================"
echo "All layers trained successfully!"
echo "================================"
echo ""
echo "Results saved to probe_results/"
echo "  - Models: probe_results/models/"
echo "  - Plots: probe_results/plots/"
echo "  - JSON: probe_results/layer_*_results.json"

