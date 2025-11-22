import os
import glob
import shutil
from sklearn.model_selection import train_test_split
import yaml

DATA_ROOT = os.path.expanduser("~/metamorphic_testing/YOLO_metamorphic_testing/training/dataset")
FSOCO_ROOT = os.path.join(DATA_ROOT, "fsoco")

OUTPUT = os.path.join(DATA_ROOT, "split")
os.makedirs(OUTPUT, exist_ok=True)

# Create directory structure
for split in ['train', 'test', 'val']:
    os.makedirs(os.path.join(OUTPUT, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, split, 'labels'), exist_ok=True)

# Collect all images from all subdirectories
images = []
subdirs = ['eufs', 'gfr', 'ka', 'mms', 'orion', 'penn', 'prc', 'prom', 
           'pwrt', 'racetech', 'rennstall', 'ugatu', 'ugr', 'umsae', 'uop', 'wfm']

for subdir in subdirs:
    img_path = os.path.join(FSOCO_ROOT, subdir, 'img')
    if os.path.exists(img_path):
        images.extend(glob.glob(os.path.join(img_path, "*.jpg")))
        images.extend(glob.glob(os.path.join(img_path, "*.png")))
        images.extend(glob.glob(os.path.join(img_path, "*.jpeg")))

images = sorted(images)
print(f"Found {len(images)} total images across all subdirectories")

# First split: 80% train, 20% temp (for val+test)
train_imgs, temp_imgs = train_test_split(
    images, 
    test_size=0.20, 
    random_state=42, 
    shuffle=True
)

# Second split: split temp into 50/50 to get 15% val and 15% test
val_imgs, test_imgs = train_test_split(
    temp_imgs, 
    test_size=0.50, 
    random_state=42, 
    shuffle=True
)

def copy_image_and_annotation(img_path, split_name):
    """Copy image and its corresponding JSON annotation to the split directory"""
    # Get the subdirectory name (e.g., 'eufs') from the image path
    rel_path = os.path.relpath(img_path, FSOCO_ROOT)
    subdir = rel_path.split(os.sep)[0]
    
    # Get image filename
    img_filename = os.path.basename(img_path)
    
    # Find corresponding annotation file
    ann_path = os.path.join(FSOCO_ROOT, subdir, 'ann', img_filename + '.json')
    
    # Destination paths
    dest_img_path = os.path.join(OUTPUT, split_name, 'images', img_filename)
    dest_ann_path = os.path.join(OUTPUT, split_name, 'labels', img_filename + '.json')
    
    # Copy image
    shutil.copy2(img_path, dest_img_path)
    
    # Copy annotation if it exists
    if os.path.exists(ann_path):
        shutil.copy2(ann_path, dest_ann_path)
    else:
        print(f"Warning: Annotation file not found for {img_path}")

# Copy files to respective directories
print("Copying training images and annotations...")
for img in train_imgs:
    copy_image_and_annotation(img, 'train')

print("Copying validation images and annotations...")
for img in val_imgs:
    copy_image_and_annotation(img, 'val')

print("Copying test images and annotations...")
for img in test_imgs:
    copy_image_and_annotation(img, 'test')

# Create dataset.yml 
yaml_data = {
    "path": DATA_ROOT,
    "train": "split/train/images",
    "val": "split/val/images",
    "test": "split/test/images",
    "nc": 6,
    "names": {
        0: "yellow_cone",
        1: "orange_cone",
        2: "large_orange_cone",
        3: "blue_cone",
        4: "other_cone",
        5: "unknown_cone",
    }
}

with open(os.path.join(OUTPUT, "dataset.yml"), "w") as f:
    yaml.dump(yaml_data, f)

print(f"Data split complete:")
print(f"  Training:   {len(train_imgs)} images ({len(train_imgs)/len(images)*100:.1f}%)")
print(f"  Validation: {len(val_imgs)} images ({len(val_imgs)/len(images)*100:.1f}%)")
print(f"  Testing:    {len(test_imgs)} images ({len(test_imgs)/len(images)*100:.1f}%)")
print(f"  Total:      {len(images)} images")
print(f"  Dataset config saved to: {os.path.join(OUTPUT, 'dataset.yml')}")