# ============================================================
# IMPORTS
# ============================================================
import os
import cv2
import numpy as np
import kagglehub
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import shutil
from preprocessing_funcs import crop_and_resize_face
from filter_closed_eyes import filter_closed_eyes
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reproducibility
RANDOM_SEED = 42

# Face extraction settings
FRAME_INTERVAL = 20          # Extract face every 20 frames (reduce redundancy)
TARGET_SIZE = (224, 224)     # Resize faces to 224x224 for model input

# Directory paths
TMP_DIR = os.path.join(ROOT_DIR, "tmp_faces")
FILTERED_DIR = os.path.join(ROOT_DIR, "filtered_faces")
FINAL_DIR = os.path.join(ROOT_DIR, "daisee_dataset")

# Dataset balancing
MAX_PER_CLASS = 4000         # Maximum images per class (0 or 1)

# Train/Validation/Test split ratios
TRAIN_RATIO = 0.8            # 80% for training
VAL_RATIO = 0.1              # 10% for Validation
TEST_RATIO = 0.1             # 10% for testing

# Set random seed for reproducible splits
np.random.seed(RANDOM_SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_daisee():
    """
    Download DAiSEE dataset from Kaggle.
    Returns the path to the dataset directory.
    """
    path = kagglehub.dataset_download("olgaparfenova/daisee")
    if os.path.exists(os.path.join(path, "DAiSEE")):
        return os.path.join(path, "DAiSEE")
    return path


def load_labels(label_dir):
    """
    Load engagement labels from all CSV files in the Labels directory.

    DAiSEE provides engagement scores 0-3 for each video clip:
    - 0, 1: Not Engaged
    - 2, 3: Engaged

    Returns:
        DataFrame with ClipID and Engagement columns
    """
    dfs = []
    for csv in Path(label_dir).glob("*.csv"):
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()
        dfs.append(df[["ClipID", "Engagement"]])
    return pd.concat(dfs, ignore_index=True)


# ============================================================
# STEP 1: EXTRACT FACES FROM VIDEOS
# ============================================================

def extract_all_faces():
    """
    Extract face images from all DAiSEE video clips.

    Process:
    1. Load video files and engagement labels
    2. For each video, extract faces every FRAME_INTERVAL frames
    3. Use YOLO to detect faces in frames
    4. Crop and resize detected faces to 224x224
    5. Save to temporary folder organized by engagement label (0 or 1)

    This step extracts ~8000-10000 face images from the entire dataset.
    """
    print("\n========== STEP 1: EXTRACTING ALL FACES ==========")

    # Check if extraction already done
    tmp_0 = os.path.join(TMP_DIR, "0")
    tmp_1 = os.path.join(TMP_DIR, "1")
    if os.path.exists(tmp_0) and os.path.exists(tmp_1):
        count_0 = len([f for f in os.listdir(tmp_0) if f.endswith('.jpg')])
        count_1 = len([f for f in os.listdir(tmp_1) if f.endswith('.jpg')])
        if count_0 > 1000 and count_1 > 1000:
            print(f"Found existing extracted faces: class 0={count_0}, class 1={count_1}")
            print("Skipping extraction step. Delete tmp_faces/ to re-extract.")
            print("="*60)
            return

    # Get DAiSEE dataset path
    root = get_daisee()
    dataset_dir = os.path.join(root, "DataSet")
    label_dir = os.path.join(root, "Labels")
    labels_df = load_labels(label_dir)

    # Load YOLO face detector (verbose=False to suppress prints)
    detector = YOLO("../yolov8n-face.pt")
    detector.verbose = False  # Suppress YOLO output

    # Create output folders for binary classification
    os.makedirs(os.path.join(TMP_DIR, "0"), exist_ok=True)  # Not Engaged
    os.makedirs(os.path.join(TMP_DIR, "1"), exist_ok=True)  # Engaged

    # Process each split (Train, Validation, Test)
    for split in ["Train", "Validation", "Test"]:
        split_path = os.path.join(dataset_dir, split)
        videos = list(Path(split_path).rglob("*.avi"))
        print(f"\nProcessing {split}: {len(videos)} videos")

        faces_extracted_this_split = 0
        videos_with_errors = 0

        for video in tqdm(videos, desc=f"Extracting faces [{split}]", ncols=80):
            # Get engagement label for this video
            row = labels_df[labels_df["ClipID"] == video.name]
            if row.empty:
                continue

            # Convert 4-level engagement (0-3) to binary (0-1)
            eng4 = int(row.iloc[0]["Engagement"])
            eng_bin = 0 if eng4 <= 1 else 1  # 0,1 -> Not Engaged; 2,3 -> Engaged

            try:
                # Open video and process frames
                cap = cv2.VideoCapture(str(video))
                if not cap.isOpened():
                    videos_with_errors += 1
                    continue

                frame_i = 0
                faces_in_video = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Extract face every FRAME_INTERVAL frames (reduce redundancy)
                    if frame_i % FRAME_INTERVAL == 0:
                        try:
                            # Detect face with YOLO (verbose=False suppresses output)
                            res = detector(frame, verbose=False)
                            if len(res) and len(res[0].boxes):
                                # Get bounding box of first detected face
                                b = res[0].boxes[0]
                                x1, y1, x2, y2 = map(int, b.xyxy[0])

                                # Crop and resize face
                                face = crop_and_resize_face(frame, x1, y1, x2, y2, target_size=TARGET_SIZE)
                                if face is not None:
                                    # Save face image to appropriate class folder
                                    out_dir = os.path.join(TMP_DIR, str(eng_bin))
                                    fname = f"{eng_bin}_{len(os.listdir(out_dir)):06d}.jpg"
                                    cv2.imwrite(os.path.join(out_dir, fname), face)
                                    faces_in_video += 1
                                    faces_extracted_this_split += 1
                        except Exception:
                            # Skip corrupted frames silently
                            pass

                    frame_i += 1

                cap.release()

            except Exception:
                # Skip corrupted videos
                videos_with_errors += 1
                continue

        # Print summary for this split
        count_0 = len([f for f in os.listdir(os.path.join(TMP_DIR, "0")) if f.endswith('.jpg')])
        count_1 = len([f for f in os.listdir(os.path.join(TMP_DIR, "1")) if f.endswith('.jpg')])
        print(f"\n{split} complete: Extracted {faces_extracted_this_split} faces")
        print(f"  Total so far - Class 0: {count_0}, Class 1: {count_1}")
        if videos_with_errors > 0:
            print(f"  Skipped {videos_with_errors} corrupted/unreadable videos")

    print("\n" + "="*60)
    print(f"EXTRACTION COMPLETE!")
    print("="*60)
    count_0 = len([f for f in os.listdir(os.path.join(TMP_DIR, "0")) if f.endswith('.jpg')])
    count_1 = len([f for f in os.listdir(os.path.join(TMP_DIR, "1")) if f.endswith('.jpg')])
    print(f"Total extracted faces:")
    print(f"  Class 0 (Not Engaged): {count_0}")
    print(f"  Class 1 (Engaged): {count_1}")
    print(f"Saved to: {TMP_DIR}")
    print("="*60)


# ============================================================
# STEP 3: BALANCE CLASSES & SPLIT DATASET
# ============================================================

def balance_and_split(input_dir, output_dir):
    """
    Balance class distribution and split into Train/Validation/Test sets.

    Process:
    1. Load all filtered face images
    2. Balance classes by limiting each to MAX_PER_CLASS images
    3. Randomly shuffle and split: 80% Train, 10% Validation, 10% Test
    4. Copy images to final organized folder structure

    This ensures:
    - Equal representation of both classes (no class imbalance)
    - Consistent train/val/test splits for reproducible experiments
    - Final dataset ready for training
    """
    print("\n========== STEP 3: BALANCE + SPLIT ==========")

    # Load all image filenames for each class
    class_files = {
        "0": sorted(os.listdir(os.path.join(input_dir, "0"))),
        "1": sorted(os.listdir(os.path.join(input_dir, "1")))
    }

    # Balance classes by limiting to MAX_PER_CLASS
    balanced = {}
    for label in ["0", "1"]:
        files = class_files[label]
        np.random.shuffle(files)  # Randomize selection
        balanced[label] = files[:MAX_PER_CLASS]
        print(f"Class {label}: {len(files)} images → balanced to {len(balanced[label])}")

    # Create output folder structure
    for split in ["Train", "Validation", "Test"]:
        for lbl in ["0", "1"]:
            os.makedirs(os.path.join(output_dir, split, lbl), exist_ok=True)

    # Split each class into train/val/test
    for label in ["0", "1"]:
        files = balanced[label]
        n = len(files)

        # Calculate split sizes
        n_train = int(TRAIN_RATIO * n)
        n_val = int(VAL_RATIO * n)

        # Split file lists
        train_f = files[:n_train]
        val_f = files[n_train:n_train+n_val]
        test_f = files[n_train+n_val:]

        # Copy files to respective folders
        def cp(lst, split):
            out = os.path.join(output_dir, split, label)
            src_root = os.path.join(input_dir, label)
            for fname in lst:
                shutil.copy2(os.path.join(src_root, fname), os.path.join(out, fname))

        cp(train_f, "Train")
        cp(val_f, "Validation")
        cp(test_f, "Test")

        print(f"Class {label}: Train={len(train_f)}, Val={len(val_f)}, Test={len(test_f)}")

    print("\nDONE: Final dataset saved to:", output_dir)


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

if __name__ == "__main__":
    """
    Run the complete preprocessing pipeline:

    Step 1: Extract faces from all DAiSEE videos
            - Detects faces in video frames using YOLO
            - Saves to tmp_faces/ folder

    Step 2: Filter closed eyes
            - Removes images where subject has closed eyes (blinks)
            - Saves cleaned images to filtered_faces/ folder

    Step 3: Balance and split
            - Balances class distribution
            - Splits into Train/Validation/Test sets
            - Final output in daisee_dataset/ folder
    """

    # Run the three-step pipeline
    extract_all_faces()                      # Step 1: Extract
    filter_closed_eyes(TMP_DIR, FILTERED_DIR)  # Step 2: Filter
    balance_and_split(FILTERED_DIR, FINAL_DIR)  # Step 3: Balance & Split

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Final dataset ready at: {FINAL_DIR}")
    print("You can now train your models using this dataset.")
    print("="*60)
