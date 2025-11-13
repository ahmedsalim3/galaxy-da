#!/bin/bash
#
# run_experiment.sh <config_path> [more_configs...] | -f <config_folder>
# Examples:
# ./run_experiment.sh configs/baseline.yml
# ./run_experiment.sh configs/a.yml configs/b.yml
# ./run_experiment.sh -f configs/

if [ "$1" = "-f" ]; then
    CONFIGS=$(ls "$2"/*.yml)
else
    CONFIGS="$@"
fi

if [ -z "$CONFIGS" ]; then
    echo "Usage: $0 <config_path> [more_configs...] | -f <config_folder>"
    exit 1
fi

export LOG_LEVEL=10 # DEBUG mode

for CONFIG_PATH in $CONFIGS; do
    echo "========================="
    echo "Using config: $CONFIG_PATH"

    EXPERIMENT_NAME=$(grep '^experiment_name:' "$CONFIG_PATH" | awk -F': ' '{print $2}' | sed 's/#.*$//' | tr -d '"' | tr -d "'" | xargs)
    METHOD=$(grep '^\s*method:' "$CONFIG_PATH" | head -n1 | awk -F': ' '{print $2}' | sed 's/#.*$//' | tr -d '"' | tr -d "'" | xargs)
    ES_METRIC=$(grep '^\s*early_stopping_metric:' "$CONFIG_PATH" | awk -F': ' '{print $2}' | sed 's/#.*$//' | tr -d '"' | tr -d "'" | xargs)

    echo "Method: $METHOD"
    echo "Early stopping metric: $ES_METRIC"
    ROOT_DIR=$(grep -A 1 '^output:' "$CONFIG_PATH" | grep 'root_dir:' | awk -F': ' '{print $2}' | sed 's/#.*$//' | tr -d '"' | tr -d "'" | xargs)

    echo "Experiment name: $EXPERIMENT_NAME"
    echo "Root directory: $ROOT_DIR"

    echo "========================="
    echo "Running training..."
    python3 scripts/run_train.py --config "$CONFIG_PATH" || continue

    CKPT_PATH="${ROOT_DIR}/${EXPERIMENT_NAME}/ckpts/${EXPERIMENT_NAME}_best_${ES_METRIC}.pt"
    echo "========================="
    echo "Running evaluation..."
    echo "Checkpoint path: $CKPT_PATH"

    [ -f "$CKPT_PATH" ] || { echo "Error: Checkpoint not found at $CKPT_PATH"; continue; }

    # OT-based methods (sinkhorn, mmd, energy) work better with UMAP
    if [[ "$METHOD" == "sinkhorn" ]] || [[ "$METHOD" == "mmd" ]] || [[ "$METHOD" == "energy" ]]; then
        EMBED_METHOD="umap"
    else
        EMBED_METHOD="tsne"
    fi

    python3 scripts/run_eval.py "$CKPT_PATH" --embed-method "$EMBED_METHOD"
done