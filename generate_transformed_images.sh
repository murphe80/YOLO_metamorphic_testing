# Generate metamorphic images using default settings
for image in data/base_images/test/images/*; do
    python utils/matias_metamorphic.py \
        --rules imgMotionBlur \
        --base_image "$image" \
        --output_path ./data/transformed_images/fsoco_test/imgMotionBlur/
done

for image in data/base_images/test/images/*; do
    python utils/matias_metamorphic.py \
        --rules imgDarkBright \
        --base_image "$image" \
        --output_path ./data/transformed_images/fsoco_test/imgDarkBright/
done

for image in data/base_images/test/images/*; do
    python utils/matias_metamorphic.py \
        --rules imgGrayscale \
        --base_image "$image" \
        --output_path ./data/transformed_images/fsoco_test/imgGrayscale/
done

for image in data/base_images/test/images/*; do
    python utils/matias_metamorphic.py \
        --rules imgResolution \
        --base_image "$image" \
        --output_path ./data/transformed_images/fsoco_test/imgResolution/
done

for image in data/base_images/test/images/*; do
    python utils/matias_metamorphic.py \
        --rules imgStretch \
        --base_image "$image" \
        --output_path ./data/transformed_images/fsoco_test/imgStretch/
done

for image in data/base_images/test/images/*; do
    python utils/matias_metamorphic.py \
        --rules imgColorWheel \
        --base_image "$image" \
        --output_path ./data/transformed_images/fsoco_test/imgColorWheel/
done