# ============================================================
# IMPORTS
# ============================================================
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import cv2

# Import all model architectures
from models import resnet18
from models import resnet18_se as se
from models import ViT as vit
from models import resnet18_gru as gru
from models import flip_invariant_resnet18 as fp

# ============================================================
# CONFIG
# ============================================================

# Get parent directory (root of project)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(ROOT_DIR, "daisee_dataset")
BATCH_SIZE = 16
IMAGE_SIZE = 224
CLASS_NAMES = ["Not Engaged", "Engaged"]

# Available models - choose which model to evaluate
MODELS = {
    0: {
        "name": "ResNet18",
        "path": os.path.join(ROOT_DIR, "weights", "Resnet18.pth"),
        "type": "frame",
        "dataset": os.path.join(ROOT_DIR, "daisee_dataset"),
        "ctor": lambda: resnet18.EngagementModel(num_classes=2),
    },
    1: {
        "name": "ResNet18+GRU",
        "path": os.path.join(ROOT_DIR, "weights", "resnet18_gru.pth"),
        "type": "sequence",
        "dataset": os.path.join(ROOT_DIR, "daisee_sequential"),
        "ctor": lambda: gru.ResNet18_GRU(),
    },
    2: {
        "name": "ResNet18+SE",
        "path": os.path.join(ROOT_DIR, "weights", "resnet18_se.pth"),
        "type": "frame",
        "dataset": os.path.join(ROOT_DIR, "daisee_dataset"),
        "ctor": lambda: se.EngagementModel(num_classes=2),
    },
    3: {
        "name": "ViT",
        "path": os.path.join(ROOT_DIR, "weights", "ViT.pth"),
        "type": "frame",
        "dataset": os.path.join(ROOT_DIR, "daisee_dataset"),
        "ctor": lambda: vit.load_model(num_classes=2),
    },
    4: {
        "name": "Flip Invariant ResNet18",
        "path": os.path.join(ROOT_DIR, "weights", "resnet18_flip_invariant.pth"),
        "type": "frame",
        "dataset": os.path.join(ROOT_DIR, "daisee_dataset"),
        "ctor": lambda: fp.FlipInvariantResNet18(num_classes=2),
    },
    5: {
        "name": "ResNet18 (No Augmentations)",
        "path": os.path.join(ROOT_DIR, "weights", "resnet18_without_augmentations.pth"),
        "type": "frame",
        "dataset": os.path.join(ROOT_DIR, "daisee_dataset"),
        "ctor": lambda: resnet18.EngagementModel(num_classes=2),
    },
}

# Select which model to evaluate (change this number: 0-4)
SELECTED_MODEL = 4


# ============================================================
# DATASET WITHOUT CSV
# ============================================================

class FolderDataset(Dataset):
    """
    Dataset for frame-based models (ResNet18, ResNet18+SE, ViT, Flip Invariant).
    Loads individual images from a folder structure.
    """
    def __init__(self, root_dir, transform=None):
        """
        root_dir/
            0/
                img1.jpg
                img2.jpg
            1/
                img1.jpg
        """
        self.samples = []
        self.transform = transform

        for label_str in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label_str)
            if not os.path.isdir(class_dir):
                continue

            label = int(label_str)

            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                if path.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = cv2.imread(path)
        if image is None:
            print(f"[WARNING] Could not read: {path}")
            # return black image so evaluation continues
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


class SequenceDataset(Dataset):
    """
    Dataset for sequence-based models (ResNet18+GRU).
    Loads sequences of frames from a folder structure.
    """
    def __init__(self, root_dir, transform=None, seq_len=10):
        """
        root_dir/
            0/
                seq001/
                    frame_0.jpg
                    frame_1.jpg
                    ...
            1/
                seq001/
                    frame_0.jpg
                    ...
        """
        self.samples = []
        self.transform = transform
        self.seq_len = seq_len

        for label_str in ["0", "1"]:
            label_dir = os.path.join(root_dir, label_str)
            if not os.path.isdir(label_dir):
                continue

            for seq_id in os.listdir(label_dir):
                seq_path = os.path.join(label_dir, seq_id)
                if not os.path.isdir(seq_path):
                    continue

                frames = sorted([
                    os.path.join(seq_path, f)
                    for f in os.listdir(seq_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])

                if len(frames) == self.seq_len:
                    self.samples.append((frames, int(label_str)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        imgs = []
        for path in frame_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"[WARNING] Could not read: {path}")
                img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img)

            imgs.append(img)

        # Stack frames into sequence: (seq_len, C, H, W)
        imgs_tensor = torch.stack(imgs, dim=0)
        return imgs_tensor, label


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model():
    """
    Evaluate the selected model on the test dataset.
    Automatically handles both frame-based and sequence-based models.
    """
    # Get selected model configuration
    model_config = MODELS[SELECTED_MODEL]
    model_name = model_config["name"]
    model_path = model_config["path"]
    model_type = model_config["type"]
    dataset_path = model_config["dataset"]
    model_ctor = model_config["ctor"]

    print("="*60)
    print(f"CONFUSION MATRIX EVALUATION")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Type: {model_type}")
    print(f"Weights: {model_path}")
    print(f"Dataset: {dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test folder
    test_dir = os.path.join(dataset_path, "Test")

    if not os.path.exists(test_dir):
        print(f"\nERROR: Test directory not found: {test_dir}")
        print("Please ensure the dataset is preprocessed correctly.")
        return

    # Transform
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create dataset based on model type
    if model_type == "sequence":
        print(f"\nLoading sequence dataset...")
        test_dataset = SequenceDataset(test_dir, eval_transform, seq_len=10)
    else:  # frame
        print(f"\nLoading frame dataset...")
        test_dataset = FolderDataset(test_dir, eval_transform)

    print(f"Loaded {len(test_dataset)} test samples.")

    if len(test_dataset) == 0:
        print(f"ERROR: No samples found in {test_dir}")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Load model using constructor from config
    print(f"\nLoading model...")
    model = model_ctor().to(device)

    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully!")

    # Collect predictions
    all_preds = []
    all_labels = []

    print("\nEvaluating model...")
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data.to(device)

            outputs = model(data)
            preds = outputs.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE}/{len(test_dataset)} samples...")

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrices
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- raw counts ---
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp1.plot(ax=ax1, cmap='Blues', colorbar=True, values_format='d')
    ax1.set_title(f'{model_name} - Counts (Acc: {accuracy * 100:.2f}%)', fontsize=12, fontweight='bold')

    # --- normalized ---
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=CLASS_NAMES)
    disp2.plot(ax=ax2, cmap='Blues', colorbar=True, values_format='.2%')
    ax2.set_title(f'{model_name} - Normalized (Acc: {accuracy * 100:.2f}%)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_model()
