# ============================================================
# IMPORTS
# ============================================================
import os
import cv2
import pandas as pd
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO
from preprocessing_funcs import detect_and_crop_face
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import kagglehub

# ============================================================
# CONFIGURATION
# ============================================================

# Get parent directory (root of project)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Helper function to get DAiSEE dataset
def get_daisee():
    """
    Download DAiSEE dataset from Kaggle.
    Returns the path to the dataset directory.
    """
    path = kagglehub.dataset_download("olgaparfenova/daisee")
    if os.path.exists(os.path.join(path, "DAiSEE")):
        return os.path.join(path, "DAiSEE")
    return path

# Get DAiSEE dataset root
DATASET_ROOT = get_daisee()
LABELS_DIR   = os.path.join(DATASET_ROOT, "Labels")
DATASET_DIR  = os.path.join(DATASET_ROOT, "DataSet")
ALL_SPLITS = ["Train", "Validation", "Test"]

# Sequence parameters
FRAMES_PER_SEQ = 10          # Number of frames per sequence (for GRU model)
TARGET_PER_CLASS = 4000      # Target sequences per class (0 or 1)
TARGET_SIZE = (224, 224)     # Resize faces to 224x224

# Train/Validation/Test split ratios
TRAIN_RATIO = 0.8            # 80% for training
VAL_RATIO = 0.1              # 10% for Validation
TEST_RATIO = 0.1             # 10% for testing

# Output directories
OUTPUT_DIR = os.path.join(ROOT_DIR, "daisee_seq") # Temporary directory for extraction
OUTPUT_FINAL = os.path.join(ROOT_DIR, "daisee_sequential") # Final organized dataset

# Initialize
os.makedirs(OUTPUT_DIR, exist_ok=True)
detector = YOLO("../yolov8n-face.pt")
detector.verbose = False  # Suppress YOLO output

# Reproducibility
random.seed(42)
np.random.seed(42)


# ============================================================
# LOAD ENGAGEMENT LABELS
# ============================================================

# Load labels from all splits and combine into single dataframe
dfs = []
for split in ALL_SPLITS:
    df_split = pd.read_csv(os.path.join(LABELS_DIR, f"{split}Labels.csv"))
    df_split["split"] = split
    dfs.append(df_split)

df = pd.concat(dfs, ignore_index=True)

# Normalize ClipID format (remove extensions, pad to 10 digits)
def normalize(x):
    x = str(x).strip()
    if "." in x:
        x = x.split(".")[0]
    return f"{int(x):010d}"

df["ClipID"] = df["ClipID"].apply(normalize)

# Convert 4-level engagement (0-3) to binary (0-1)
# 0,1 -> Not Engaged (0)
# 2,3 -> Engaged (1)
df["binary"] = df["Engagement"].apply(lambda x: 0 if x in [0,1] else 1)

# Create mapping: ClipID -> binary label
label_map = dict(zip(df["ClipID"], df["binary"]))


# ============================================================
# COLLECT VIDEO FILES
# ============================================================

# Find all video files in the dataset
videos = []
for split in ALL_SPLITS:
    videos.extend(Path(DATASET_DIR, split).rglob("*.avi"))
print(f"Total videos found: {len(videos)}")

# Separate videos by engagement label
engaged_videos = []    # Label 1
noteng_videos = []     # Label 0

for v in videos:
    vid = v.stem
    if vid not in label_map:
        continue

    if label_map[vid] == 1:
        engaged_videos.append(v)
    else:
        noteng_videos.append(v)

print(f"Engaged videos (label 1): {len(engaged_videos)}")
print(f"Not engaged videos (label 0): {len(noteng_videos)}")



# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_random_frames(path, k=10):
    """
    Extract k random frames from a video file.

    This creates temporal sequences by randomly sampling frames
    rather than taking consecutive frames. This adds variety and
    reduces redundancy in the dataset.

    Args:
        path: Path to video file
        k: Number of frames to extract (default: 10)

    Returns:
        List of k frames, or None if video has fewer than k frames
    """
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if video has enough frames
    if total < k:
        cap.release()
        return None

    # Randomly select k frame indices and sort them
    idxs = np.sort(np.random.choice(total, k, replace=False))

    # Extract the selected frames
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        frames.append(frame)

    cap.release()
    return frames if len(frames) == k else None


def save_sequence(args):
    """
    Extract and save a sequence of face images from a video.

    Process:
    1. Extract 10 random frames from video
    2. Detect and crop face in each frame using YOLO
    3. Save cropped faces as a numbered sequence

    Args:
        args: Tuple of (video_path, label_folder, seq_id)

    Returns:
        True if all 10 frames were successfully saved, False otherwise
    """
    video_path, label_folder, seq_id = args

    # Extract random frames from video
    frames = extract_random_frames(video_path, FRAMES_PER_SEQ)
    if frames is None:
        return False

    # Create directory for this sequence
    seq_dir = os.path.join(OUTPUT_DIR, label_folder, f"{seq_id:06d}")
    os.makedirs(seq_dir, exist_ok=True)

    # Process each frame: detect face, crop, and save
    saved = 0
    for j, f in enumerate(frames):
        face, found = detect_and_crop_face(detector, f, target_size=TARGET_SIZE)
        if found:
            cv2.imwrite(os.path.join(seq_dir, f"frame_{j:04d}.jpg"), face)
            saved += 1

    # Only count as success if all frames were saved
    return saved == FRAMES_PER_SEQ


def build_class_parallel(videos, label_folder, start_index, oversample=False):
    """
    Build a dataset class using parallel processing.

    This function extracts sequences from videos using multiple workers
    to speed up processing. Can oversample if there aren't enough videos
    to reach TARGET_PER_CLASS.

    Args:
        videos: List of video file paths
        label_folder: Output folder ("0" or "1")
        start_index: Starting sequence ID
        oversample: If True, randomly sample videos with replacement

    Process:
    - Uses 6 parallel workers for faster extraction
    - Continues until TARGET_PER_CLASS sequences are created
    - Shows progress every 50 sequences
    """
    if not oversample:
        videos = videos[:TARGET_PER_CLASS]

    seq_id = start_index
    tasks = []

    print(f"Starting {label_folder}: from seq {start_index}")

    # Use parallel processing with 6 workers
    with ProcessPoolExecutor(max_workers=6) as ex:

        # Submit tasks for each sequence
        while seq_id < TARGET_PER_CLASS:
            # If oversampling, randomly pick video; otherwise use in order
            vid = random.choice(videos) if oversample else videos[seq_id]

            tasks.append(ex.submit(save_sequence, (vid, label_folder, seq_id)))
            seq_id += 1

        # Track progress as tasks complete
        done_count = start_index
        for f in as_completed(tasks):
            ok = f.result()
            if ok:
                done_count += 1

            # Show progress update every 50 sequences
            if done_count % 50 == 0:
                print(f"{label_folder}: {done_count}/{TARGET_PER_CLASS}")

    print(f"Completed {label_folder}")

def split_sequences():
    """
    Split extracted sequences into Train/Validation/Test sets.

    Process:
    1. List all sequence folders for each class
    2. Split each class: 80% Train, 10% Validation, 10% Test
    3. Move sequences to final organized folder structure

    This ensures balanced splits across both classes for
    reproducible training and evaluation.
    """
    print("\n========== SPLITTING INTO TRAIN/VAL/TEST ==========")

    for label in ["0", "1"]:
        seqs = sorted(os.listdir(os.path.join(OUTPUT_DIR, label)))
        n = len(seqs)

        # Calculate split sizes
        n_train = int(TRAIN_RATIO * n)
        n_val   = int(VAL_RATIO * n)

        # Split sequence lists
        train_seqs = seqs[:n_train]
        val_seqs   = seqs[n_train:n_train+n_val]
        test_seqs  = seqs[n_train+n_val:]

        # Create output directories for each split
        for split in ["Train", "Validation", "Test"]:
            os.makedirs(os.path.join(OUTPUT_FINAL, split, label), exist_ok=True)

        # Move sequences to respective folders
        def move(split_name, items):
            for seq in items:
                src = os.path.join(OUTPUT_DIR, label, seq)
                dst = os.path.join(OUTPUT_FINAL, split_name, label, seq)
                shutil.move(src, dst)

        move("Train", train_seqs)
        move("Validation", val_seqs)
        move("Test", test_seqs)

        print(f"Label {label}: Train={len(train_seqs)}, Validation={len(val_seqs)}, Test={len(test_seqs)}")

    print("\nSplitting complete")

# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

if __name__ == "__main__":
    """
    Run the complete sequential preprocessing pipeline.
    
    Step 1: Extract sequences from engaged videos (label 1)
            - 4000 sequences without oversampling
    
    Step 2: Extract sequences from not engaged videos (label 0)
            - 4000 sequences with oversampling (fewer source videos)
    
    Step 3: Split into Train/Validation/Test (80/10/10)
    
    Total output: 8000 sequences (4000 per class)
                  Each sequence contains 10 face images
    
    Uses parallel processing for efficiency (6 workers).
    """

    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, "0"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "1"), exist_ok=True)

    # Step 1: Process engaged videos
    print("\n========== PROCESSING ENGAGED VIDEOS (label 1) ==========")
    random.shuffle(engaged_videos)
    build_class_parallel(engaged_videos,
                         label_folder="1",
                         start_index=0,
                         oversample=False)

    # Step 2: Process not engaged videos
    print("\n========== PROCESSING NOT ENGAGED VIDEOS (label 0) ==========")
    build_class_parallel(noteng_videos,
                         label_folder="0",
                         start_index=0,
                         oversample=True)

    print("\nSequence extraction complete: 8000 total sequences created")

    # Step 3: Split into train/val/test
    split_sequences()

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Final dataset ready at: {OUTPUT_FINAL}")
    print("Dataset contains 10-frame sequences for temporal models")
    print("="*60)
