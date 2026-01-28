# ============================================================
# IMPORTS
# ============================================================
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import random

# ============================================================
# CONFIGURATION
# ============================================================

# Reproducibility
SEED = 42

# Model parameters
NUM_CLASSES = 2              # Binary classification: 0=Not Engaged, 1=Engaged
IMAGE_SIZE = 224             # Input image size (224x224)

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4          # L2 regularization

# Learning rate scheduler
LR_STEP_SIZE = 8             # Reduce LR every 8 epochs
LR_GAMMA = 0.5               # Multiply LR by 0.5

# Early stopping
PATIENCE = 8                 # Stop training if no improvement for 8 epochs

# DataLoader settings
NUM_WORKERS = 4
PIN_MEMORY = True

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "daisee_dataset")
MODEL_PATH = os.path.join(ROOT_DIR, "weights", "Resnet18.pth")

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    """Seed DataLoader workers for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============================================================
# DATASET
# ============================================================

class FolderDataset(Dataset):
    """
    Load images directly from folder structure.
    Expected structure:
        root_dir/
            0/  (Not Engaged)
            1/  (Engaged)
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Load images from both class folders
        for label_str in ["0", "1"]:
            class_dir = os.path.join(root_dir, label_str)
            if not os.path.isdir(class_dir):
                continue

            label = int(label_str)

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load and convert image to RGB
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# MODEL
# ============================================================

class EngagementModel(nn.Module):
    """
    ResNet18-based engagement detection model.
    Uses pretrained ImageNet weights with dropout for regularization.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # Load pretrained ResNet18
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Replace final layer with dropout + classifier
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone.fc.in_features, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """
    Run one epoch of training or evaluation.

    Args:
        model: Neural network model
        loader: DataLoader for the dataset
        criterion: Loss function
        optimizer: Optimizer (None for evaluation)
        device: Device to run on (cpu/cuda)

    Returns:
        Average loss and accuracy for the epoch
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(SEED)

    # Create weights directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)

    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths to train/val/test folders
    train_dir = os.path.join(DATASET_DIR, "Train")
    val_dir = os.path.join(DATASET_DIR, "Validation")
    test_dir = os.path.join(DATASET_DIR, "Test")

    print(f"\nLoading dataset from: {DATASET_DIR}")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2)
    ])

    # Standard normalization for Validation/test
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create datasets from folders
    train_dataset = FolderDataset(train_dir, train_transform)
    val_dataset = FolderDataset(val_dir, eval_transform)
    test_dataset = FolderDataset(test_dir, eval_transform)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Create dataloaders with reproducible seeding
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Initialize model, loss, optimizer, and scheduler
    model = EngagementModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA
    )

    # Training loop with early stopping
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # Train for one epoch
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate on Validation set
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device
        )

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")

            # Early stopping
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best Validation accuracy: {best_val_acc:.2f}%")
                break

    # Evaluate on test set
    print("EVALUATING ON TEST SET")

    model.load_state_dict(torch.load(MODEL_PATH))
    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, None, device
    )

    print(f"\nFinal Results:")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")

