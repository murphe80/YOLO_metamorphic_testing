#!/bin/bash
# ============================================
# run_all_matias_transformations.sh
#
# Apply metamorphic image transformations to a dataset.
# Supports fsoco_sample (bundled) and test (held-out) datasets.
#
# Usage:
#   bash scripts/run_all_matias_transformations.sh [--dataset fsoco_sample|test]
#
# Outputs: data/transformed_images/<dataset>/<rule>/
# ============================================

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# ============================================
# Parse arguments
# ============================================
DATASET="fsoco_sample"  # Default to sample dataset

for arg in "$@"; do
    case $arg in
        --dataset) DATASET="${2:-fsoco_sample}"; shift ;;
        --dataset=*) DATASET="${arg#*=}" ;;
    esac
done

# ============================================
# Configuration Parameters
# ============================================

# Set paths based on dataset
if [ "$DATASET" = "fsoco_sample" ]; then
    BASE_IMAGES_DIR="data/base_images/fsoco_sample/segmented_images"
    OUTPUT_BASE_DIR="data/transformed_images/fsoco_sample"
elif [ "$DATASET" = "test" ]; then
    BASE_IMAGES_DIR="data/base_images/test/images"
    OUTPUT_BASE_DIR="data/transformed_images/fsoco_test"
else
    # Custom dataset path
    BASE_IMAGES_DIR="data/base_images/$DATASET/images"
    OUTPUT_BASE_DIR="data/transformed_images/$DATASET"
fi

# Rule configurations (JSON format; parallel arrays for bash 3.2 on macOS)
RULE_NAMES=(
    imgDarkBright
    imgResolution
    imgMotionBlur
    imgStretch
    imgGrayscale
    imgColorWheel
)
RULE_CONFIGS=(
    '{"brightness_constant": -20, "brightness_multiplier": 0.2, "gamma": 1, "method": "gamma"}'
    '{"size_ratio": 0.6}'
    '{"kernel_size": 12, "orientation": 45}'
    '{"height_ratio": 1.5, "width_ratio": 1.5}'
    '{}'
    '{"hue_rotation": 15}'
)

# ============================================
# Validate input directory
# ============================================

if [ ! -d "$BASE_IMAGES_DIR" ]; then
    echo "Error: Base images directory not found: $BASE_IMAGES_DIR"
    echo "Make sure the dataset exists or use --dataset fsoco_sample"
    exit 1
fi

# ============================================
# Generate metamorphic images
# ============================================

echo "======================================"
echo "Metamorphic Image Transformations"
echo "======================================"
echo "Dataset: $DATASET"
echo "Input: $BASE_IMAGES_DIR"
echo "Output base: $OUTPUT_BASE_DIR"
echo ""

# Process each rule
for i in "${!RULE_NAMES[@]}"; do
    rule="${RULE_NAMES[$i]}"
    config="${RULE_CONFIGS[$i]}"
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
