#!/bin/bash
# Train probes for all layers with proper file-wise splitting

ACTIVATIONS_DIR="${1:-/workspace/activations/Qwen3-0.6B}"
MAX_FILES="${2:-30}"  # Limit files per temperature for faster training
TOKEN_EVERY_N="${3:-10}"
CUSTOM_LAYER_SPEC="${4:-}"

echo "Training probes for all layers"
echo "Activations dir: $ACTIVATIONS_DIR"
echo "Max files per temp: $MAX_FILES"
echo "Token sampling: every $TOKEN_EVERY_N"
if [ -n "$CUSTOM_LAYER_SPEC" ]; then
    echo "Custom layer selection: $CUSTOM_LAYER_SPEC"
fi
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
BATCH_SIZE=2

# Build list of target layers (optionally custom)
declare -a TARGET_LAYERS=()
declare -A LAYER_SEEN=()

if [ -n "$CUSTOM_LAYER_SPEC" ]; then
    IFS=',' read -ra RAW_LAYERS <<< "$CUSTOM_LAYER_SPEC"
    for entry in "${RAW_LAYERS[@]}"; do
        entry="${entry//[[:space:]]/}"
        if [ -z "$entry" ]; then
            continue
        fi

        if [[ "$entry" =~ ^[0-9]+-[0-9]+$ ]]; then
            start="${entry%-*}"
            end="${entry#*-}"
            if [ "$start" -gt "$end" ]; then
                echo "ERROR: Invalid layer range '$entry' (start > end)"
                exit 1
            fi
            for layer in $(seq "$start" "$end"); do
                if [ "$layer" -lt 0 ] || [ "$layer" -ge "$NUM_LAYERS" ]; then
                    echo "ERROR: Layer index $layer out of bounds (0-$((NUM_LAYERS - 1)))"
                    exit 1
                fi
                if [ -z "${LAYER_SEEN[$layer]}" ]; then
                    TARGET_LAYERS+=("$layer")
                    LAYER_SEEN["$layer"]=1
                fi
            done
        elif [[ "$entry" =~ ^[0-9]+$ ]]; then
            layer="$entry"
            if [ "$layer" -lt 0 ] || [ "$layer" -ge "$NUM_LAYERS" ]; then
                echo "ERROR: Layer index $layer out of bounds (0-$((NUM_LAYERS - 1)))"
                exit 1
            fi
            if [ -z "${LAYER_SEEN[$layer]}" ]; then
                TARGET_LAYERS+=("$layer")
                LAYER_SEEN["$layer"]=1
            fi
        else
            echo "ERROR: Unable to parse layer entry '$entry'. Use comma-separated indices or ranges (e.g., 1,5,9 or 2-6)."
            exit 1
        fi
    done
else
    for layer in $(seq 0 $((NUM_LAYERS - 1))); do
        TARGET_LAYERS+=("$layer")
    done
fi

if [ "${#TARGET_LAYERS[@]}" -eq 0 ]; then
    echo "ERROR: No layers selected for training."
    exit 1
fi

echo "Total layers to train: ${#TARGET_LAYERS[@]}"

TOTAL_GROUPS=$(( (${#TARGET_LAYERS[@]} + BATCH_SIZE - 1) / BATCH_SIZE ))
GROUP_INDEX=1

for ((idx=0; idx<${#TARGET_LAYERS[@]}; idx+=BATCH_SIZE)); do
    group_layers=("${TARGET_LAYERS[@]:idx:BATCH_SIZE}")
    group_str=$(printf "%s," "${group_layers[@]}")
    group_str=${group_str%,}

    echo "================================"
    echo "Training group $GROUP_INDEX/$TOTAL_GROUPS (layers: $group_str)"
    echo "================================"

    python3 utils/train_probe_v2.py \
        --activations_dir "$ACTIVATIONS_DIR" \
        --layers "$group_str" \
        --token_every_n "$TOKEN_EVERY_N" \
        --max_files_per_temp "$MAX_FILES" \
        --device cuda \
        --preload_to_gpu \
        --epochs 1000 \
        --patience 50 \
        --output_dir "probe_results"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to train layer group $group_str"
        exit 1
    fi

    GROUP_INDEX=$((GROUP_INDEX + 1))
    echo ""
done

echo "================================"
echo "All selected layers trained successfully!"
echo "================================"
echo ""
echo "Results saved to probe_results/"
echo "  - Models: probe_results/models/"
echo "  - Plots: probe_results/plots/"
echo "  - JSON: probe_results/layer_*_results.json"

