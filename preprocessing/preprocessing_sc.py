# ============================================================
# IMPORTS
# ============================================================
import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import preprocessing_funcs as ppf
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Directory paths
ORIGIN_DIR = os.path.join(ROOT_DIR, "student dataset")
INTERMEDIATE_DIR = os.path.join(ROOT_DIR, "processed_sc")
FINAL_DIR = os.path.join(ROOT_DIR, "processed_sc_ready")

# Dataset balancing
MAX_PER_CLASS = 1000                     # Maximum images per class (0 or 1)

# Train/Validation/Test split ratios
TRAIN_RATIO = 0.8                        # 80% for training
VAL_RATIO = 0.1                          # 10% for Validation
TEST_RATIO = 0.1                         # 10% for testing

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# STEP 1: CROP FACES FROM RAW IMAGES
# ============================================================

def preprocess_images():
    """
    Extract and crop faces from raw student images.

    Process:
    1. Load images from Engaged and Not Engaged folders
    2. Detect faces using YOLO
    3. Crop and resize faces to 224x224
    4. Save cropped faces to intermediate directory

    This ensures all images have consistent face regions and size.
    """
    print("\n========== STEP 1: CROPPING FACES ==========")

    # Map folder names to binary labels
    class_map = {"Engaged": 1, "Not Engaged": 0}

    # Load YOLO face detector (verbose=False to suppress prints)
    detector = YOLO("../yolov8n-face.pt")
    detector.verbose = False  # Suppress YOLO output

    # Create output folders for each class
    for lbl in [0, 1]:
        os.makedirs(os.path.join(INTERMEDIATE_DIR, str(lbl)), exist_ok=True)

    # Process each class folder
    for class_name, label in class_map.items():
        source_dir = os.path.join(ORIGIN_DIR, class_name)
        target_dir = os.path.join(INTERMEDIATE_DIR, str(label))

        # Check if folder exists
        if not os.path.exists(source_dir):
            print(f"Warning: Missing folder {source_dir}")
            continue

        images = sorted(os.listdir(source_dir))
        print(f"\nProcessing {class_name} ({label}): {len(images)} images")

        # Process each image in the folder
        for i, fname in enumerate(tqdm(images, desc=f"Cropping faces [{class_name}]", ncols=80)):
            src_path = os.path.join(source_dir, fname)

            # Read image
            frame = cv2.imread(src_path)
            if frame is None:
                continue

            # Detect and crop face
            cropped, found = ppf.detect_and_crop_face(detector, frame)
            if not found:
                continue

            # Save cropped face image
            out_name = f"{label}_{len(os.listdir(target_dir)):06d}.jpg"
            out_path = os.path.join(target_dir, out_name)
            cv2.imwrite(out_path, cropped)

    print("\nFace cropping complete. Saved to:", INTERMEDIATE_DIR)


# ============================================================
# STEP 2: BALANCE CLASSES & SPLIT DATASET
# ============================================================

def balance_and_split():
    """
    Balance class distribution and split into Train/Validation/Test sets.

    Process:
    1. Load all cropped face images
    2. Balance classes by limiting to MAX_PER_CLASS images per class
    3. Randomly shuffle and split: 80% Train, 10% Validation, 10% Test
    4. Copy images to final organized folder structure

    This ensures equal representation of both classes and
    consistent splits for reproducible training.
    """
    print("\n========== STEP 2: BALANCING & SPLITTING ==========")

    # Collect all cropped face images for each class
    images_by_class = {
        0: sorted(os.listdir(os.path.join(INTERMEDIATE_DIR, "0"))),
        1: sorted(os.listdir(os.path.join(INTERMEDIATE_DIR, "1")))
    }

    # Balance classes by limiting to MAX_PER_CLASS
    balanced = {}
    for label in [0, 1]:
        files = images_by_class[label]
        np.random.shuffle(files)  # Randomize selection
        limited = files[:MAX_PER_CLASS]
        balanced[label] = limited
        print(f"Class {label}: {len(files)} images balanced to {len(limited)}")

    # Create final folder structure
    for split in ["Train", "Validation", "Test"]:
        for lbl in [0, 1]:
            os.makedirs(os.path.join(FINAL_DIR, split, str(lbl)), exist_ok=True)

    # Split each class into train/val/test
    for label in [0, 1]:
        files = balanced[label]
        n = len(files)

        # Calculate split sizes
        n_train = int(TRAIN_RATIO * n)
        n_val = int(VAL_RATIO * n)

        # Split file lists
        train_f = files[:n_train]
        val_f   = files[n_train:n_train+n_val]
        test_f  = files[n_train+n_val:]

        # Copy files to respective folders
        def copy_to(split, flist):
            out_dir = os.path.join(FINAL_DIR, split, str(label))
            for i, fname in enumerate(flist):
                src = os.path.join(INTERMEDIATE_DIR, str(label), fname)
                dst = os.path.join(out_dir, f"{label}_{i:05d}.jpg")
                shutil.copy2(src, dst)

        copy_to("Train", train_f)
        copy_to("Validation", val_f)
        copy_to("Test", test_f)

        print(f"Class {label}: Train={len(train_f)}, Validation={len(val_f)}, Test={len(test_f)}")

    print("\nFinal dataset created at:", FINAL_DIR)


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

if __name__ == "__main__":
    """
    Run the preprocessing pipeline for student-collected dataset.
    
    Step 1: Crop faces from raw images
            - Detects faces using YOLO
            - Crops and resizes to 224x224
            - Saves to processed_sc/ folder
    
    Step 2: Balance and split
            - Balances class distribution
            - Splits into Train/Validation/Test sets
            - Final output in processed_sc_ready/ folder
    
    Note: No closed-eye filtering needed since we manually
          collected these images (no video frames with blinks)
    """

    # Run the two-step pipeline
    preprocess_images()      # Step 1: Crop faces
    balance_and_split()      # Step 2: Balance & Split

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Final dataset ready at: {FINAL_DIR}")
    print("Dataset is organized in folders (no CSV file needed)")
    print("="*60)
