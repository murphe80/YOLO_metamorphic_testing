import os
import glob
import shutil
from sklearn.model_selection import KFold
import yaml

DATA_ROOT = os.path.expanduser("~/metamorphic_testing/YOLO_metamorphic_testing/training/dataset")
SPLIT_ROOT = os.path.join(DATA_ROOT, "split")
KFOLD_ROOT = os.path.join(DATA_ROOT, "fsoco", "kfold")

# Number of folds
N_FOLDS = 5

# Create kfold root directory
os.makedirs(KFOLD_ROOT, exist_ok=True)

# Collect all images from train and val splits (combine them for k-fold)
train_images = []
val_images = []

train_img_dir = os.path.join(SPLIT_ROOT, "train", "images")
val_img_dir = os.path.join(SPLIT_ROOT, "val", "images")

if os.path.exists(train_img_dir):
    train_images = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg")) +
                         glob.glob(os.path.join(train_img_dir, "*.png")) +
                         glob.glob(os.path.join(train_img_dir, "*.jpeg")))

if os.path.exists(val_img_dir):
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")) +
                       glob.glob(os.path.join(val_img_dir, "*.png")) +
                       glob.glob(os.path.join(val_img_dir, "*.jpeg")))

# Combine train and val for k-fold cross-validation
all_images = train_images + val_images
print(f"Found {len(train_images)} training images and {len(val_images)} validation images")
print(f"Total images for k-fold: {len(all_images)}")

if len(all_images) == 0:
    print("Error: No images found in train/val directories. Please run split.py first.")
    exit(1)

# Create KFold splitter
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Process each fold
for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(all_images), 1):
    print(f"\nProcessing fold {fold_idx}/{N_FOLDS}...")
    
    # Create fold directory structure
    fold_dir = os.path.join(KFOLD_ROOT, f"fold{fold_idx}")
    fold_train_dir = os.path.join(fold_dir, "train", "images")
    fold_train_labels_dir = os.path.join(fold_dir, "train", "labels")
    fold_val_dir = os.path.join(fold_dir, "val", "images")
    fold_val_labels_dir = os.path.join(fold_dir, "val", "labels")
    
    os.makedirs(fold_train_dir, exist_ok=True)
    os.makedirs(fold_train_labels_dir, exist_ok=True)
    os.makedirs(fold_val_dir, exist_ok=True)
    os.makedirs(fold_val_labels_dir, exist_ok=True)
    
    # Get images for this fold
    fold_train_images = [all_images[i] for i in train_indices]
    fold_val_images = [all_images[i] for i in val_indices]
    
    print(f"  Train: {len(fold_train_images)} images")
    print(f"  Val: {len(fold_val_images)} images")
    
    # Copy training images and labels
    for img_path in fold_train_images:
        img_filename = os.path.basename(img_path)
        label_filename = img_filename + '.json'
        
        # Source label path (from split directory)
        src_label_path = os.path.join(SPLIT_ROOT, "train" if img_path in train_images else "val", "labels", label_filename)
        
        # Destination paths
        dest_img_path = os.path.join(fold_train_dir, img_filename)
        dest_label_path = os.path.join(fold_train_labels_dir, label_filename)
        
        # Copy image
        shutil.copy2(img_path, dest_img_path)
        
        # Copy label if it exists
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dest_label_path)
        else:
            print(f"    Warning: Label not found for {img_filename}")
    
    # Copy validation images and labels
    for img_path in fold_val_images:
        img_filename = os.path.basename(img_path)
        label_filename = img_filename + '.json'
        
        # Source label path (from split directory)
        src_label_path = os.path.join(SPLIT_ROOT, "train" if img_path in train_images else "val", "labels", label_filename)
        
        # Destination paths
        dest_img_path = os.path.join(fold_val_dir, img_filename)
        dest_label_path = os.path.join(fold_val_labels_dir, label_filename)
        
        # Copy image
        shutil.copy2(img_path, dest_img_path)
        
        # Copy label if it exists
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dest_label_path)
        else:
            print(f"    Warning: Label not found for {img_filename}")
    
    # Create YAML file for this fold
    yaml_data = {
        "path": DATA_ROOT,
        "train": f"fsoco/kfold/fold{fold_idx}/train/images",
        "val": f"fsoco/kfold/fold{fold_idx}/val/images",
        "test": "split/test/images",  # Keep test set separate
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
    
    yaml_path = os.path.join(fold_dir, f"fold{fold_idx}.yml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)
    
    print(f"  Created {yaml_path}")

print(f"\nK-fold preparation complete!")
print(f"Created {N_FOLDS} folds in {KFOLD_ROOT}")
print(f"Each fold has its own train/val split and fold{i}.yml configuration file")

