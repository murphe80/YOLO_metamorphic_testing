#!/bin/bash
# ============================================
# model_registry.sh
#
# Shell helper functions for accessing model registry.
# Source this file in your bash scripts to use the functions.
#
# Usage:
#   source utils/model_registry.sh
#
#   # Get all model names
#   MODEL_NAMES=$(get_model_names)
#
#   # Get model config field
#   WEIGHTS=$(get_model_field "yolov5" "weights")
#   SCRIPT=$(get_model_field "yolov8" "inference_script")
#
#   # Iterate over all models
#   for model in $(get_model_names); do
#     echo "Processing $model..."
#     weights=$(get_model_field "$model" "weights")
#     script=$(get_model_field "$model" "inference_script")
#   done
# ============================================

# Get the directory where this script is located
# Handle both sourced and executed contexts
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
MODEL_CONFIG="$SCRIPT_DIR/../config/models.json"

# Fallback: if config doesn't exist, try relative to current directory
if [ ! -f "$MODEL_CONFIG" ]; then
    MODEL_CONFIG="config/models.json"
fi

# Check if jq is available, otherwise use python
if command -v jq &> /dev/null; then
    USE_JQ=true
else
    USE_JQ=false
fi

# Get list of model names (space-separated)
get_model_names() {
    local enabled_only="${1:-true}"

    if [ "$USE_JQ" = true ]; then
        if [ "$enabled_only" = "true" ]; then
            jq -r '.models[] | select(.enabled == true) | .name' "$MODEL_CONFIG" | tr '\n' ' '
        else
            jq -r '.models[].name' "$MODEL_CONFIG" | tr '\n' ' '
        fi
    else
        python3 -c "
import json, sys
with open('$MODEL_CONFIG') as f:
    config = json.load(f)
    models = config['models']
    if '$enabled_only' == 'true':
        models = [m for m in models if m.get('enabled', True)]
    print(' '.join(m['name'] for m in models))
"
    fi
}

# Get list of model versions (space-separated)
get_model_versions() {
    local enabled_only="${1:-true}"

    if [ "$USE_JQ" = true ]; then
        if [ "$enabled_only" = "true" ]; then
            jq -r '.models[] | select(.enabled == true) | .version' "$MODEL_CONFIG" | tr '\n' ' '
        else
            jq -r '.models[].version' "$MODEL_CONFIG" | tr '\n' ' '
        fi
    else
        python3 -c "
import json, sys
with open('$MODEL_CONFIG') as f:
    config = json.load(f)
    models = config['models']
    if '$enabled_only' == 'true':
        models = [m for m in models if m.get('enabled', True)]
    print(' '.join(m['version'] for m in models))
"
    fi
}

# Get a specific field from a model config
# Usage: get_model_field "yolov5" "weights"
get_model_field() {
    local model_name="$1"
    local field="$2"

    if [ "$USE_JQ" = true ]; then
        jq -r ".models[] | select(.name == \"$model_name\") | .$field // empty" "$MODEL_CONFIG"
    else
        python3 -c "
import json, sys
with open('$MODEL_CONFIG') as f:
    config = json.load(f)
    for model in config['models']:
        if model['name'] == '$model_name':
            # Handle nested fields like 'color.rgb'
            value = model
            for key in '$field'.split('.'):
                value = value.get(key, '')
                if not value:
                    break
            print(value)
            break
"
    fi
}

# Get entire model config as JSON
# Usage: get_model_config "yolov5"
get_model_config() {
    local model_name="$1"

    if [ "$USE_JQ" = true ]; then
        jq -r ".models[] | select(.name == \"$model_name\")" "$MODEL_CONFIG"
    else
        python3 -c "
import json, sys
with open('$MODEL_CONFIG') as f:
    config = json.load(f)
    for model in config['models']:
        if model['name'] == '$model_name':
            print(json.dumps(model, indent=2))
            break
"
    fi
}

# Get model name by version
# Usage: get_model_by_version "v5"
get_model_by_version() {
    local version="$1"

    if [ "$USE_JQ" = true ]; then
        jq -r ".models[] | select(.version == \"$version\") | .name" "$MODEL_CONFIG"
    else
        python3 -c "
import json
with open('$MODEL_CONFIG') as f:
    config = json.load(f)
    for model in config['models']:
        if model['version'] == '$version':
            print(model['name'])
            break
"
    fi
}

# Validate that version argument is valid
# Usage: validate_model_version "v5" || { echo "Invalid model"; exit 1; }
validate_model_version() {
    local version="$1"
    local valid_versions
    valid_versions=$(get_model_versions)

    for v in $valid_versions; do
        if [ "$v" = "$version" ]; then
            return 0
        fi
    done
    return 1
}

# Get default inference parameter
# Usage: CONF_THRES=$(get_default_param "conf_thres")
get_default_param() {
    local param="$1"

    if [ "$USE_JQ" = true ]; then
        jq -r ".default_inference_params.$param // empty" "$MODEL_CONFIG"
    else
        python3 -c "
import json
with open('$MODEL_CONFIG') as f:
    config = json.load(f)
    print(config.get('default_inference_params', {}).get('$param', ''))
"
    fi
}

# Export functions for use in scripts that source this file
export -f get_model_names
export -f get_model_versions
export -f get_model_field
export -f get_model_config
export -f get_model_by_version
export -f validate_model_version
export -f get_default_param
