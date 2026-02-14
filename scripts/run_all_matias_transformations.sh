#!/bin/bash
# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# ============================================
# Configuration Parameters
# ============================================

# Input directory
BASE_IMAGES_DIR="data/base_images/test/images"
OUTPUT_BASE_DIR="data/transformed_images/fsoco_test"

# Rule configurations (JSON format)
declare -A RULE_CONFIGS
RULE_CONFIGS[imgDarkBright]='{"brightness_constant": -20, "brightness_multiplier": 0.2, "gamma": 1, "method": "gamma"}'
RULE_CONFIGS[imgResolution]='{"size_ratio": 0.6}'
RULE_CONFIGS[imgMotionBlur]='{"kernel_size": 12, "orientation": 45}'
RULE_CONFIGS[imgStretch]='{"height_ratio": 1.5, "width_ratio": 1.5}'
RULE_CONFIGS[imgGrayscale]='{}'
RULE_CONFIGS[imgColorWheel]='{"hue_rotation": 15}'

# ============================================
# Generate metamorphic images
# ============================================


echo "Input: $BASE_IMAGES_DIR"
echo "Output base: $OUTPUT_BASE_DIR"
echo ""

# Process each rule
for rule in "${!RULE_CONFIGS[@]}"; do
    config="${RULE_CONFIGS[$rule]}"
    output_dir="$OUTPUT_BASE_DIR/$rule"
    
    echo "======================================"
    echo "Applying transformation: $rule"
    echo "Configuration: $config"
    echo "Output directory: $output_dir"
    echo "======================================"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Process each image
    img_count=0
    for image in "$BASE_IMAGES_DIR"/*; do
        [ ! -f "$image" ] && continue
        
        echo "Processing: $(basename "$image")"
        python utils/matias_metamorphic.py \
            --rule "$rule" \
            --config "$config" \
            --base_image "$image" \
            --output_path "$output_dir" \
            --save-config
        
        if [ $? -eq 0 ]; then
            ((img_count++))
        else
            echo "  ✗ Failed to process $(basename "$image")"
        fi
    done
    
    echo "✓ Processed $img_count images for $rule"
    echo ""
done

echo "Summary:"
for rule in "${!RULE_CONFIGS[@]}"; do
    output_dir="$OUTPUT_BASE_DIR/$rule"
    if [ -d "$output_dir" ]; then
        num_files=$(find "$output_dir" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
        echo "  - $rule: $num_files files in $output_dir"
        if [ -f "$output_dir/configs.json" ]; then
            echo "    Config: $output_dir/configs.json"
        fi
    fi
done
