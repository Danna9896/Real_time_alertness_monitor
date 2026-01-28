# ============================================================
# IMPORTS
# ============================================================
import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

# Reproducibility
SEED = 42

# Model parameters
NUM_CLASSES = 2              # Binary classification: 0=Not Engaged, 1=Engaged
SEQ_LEN = 10                 # Number of frames per sequence (temporal context)
IMAGE_SIZE = 224             # Input image size

# Training parameters
BATCH_SIZE = 8               # Smaller batch size for sequences (memory intensive)
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4          # L2 regularization

# Learning rate scheduler
LR_STEP_SIZE = 8             # Reduce LR every 8 epochs
LR_GAMMA = 0.5               # Multiply LR by 0.5

# Early stopping
PATIENCE = 8                 # Stop if no improvement for 8 epochs

# DataLoader settings
NUM_WORKERS = 4
PIN_MEMORY = True

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "weights", "resnet18_gru.pth")

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

set_seed(SEED)


# ============================================================
# SEQUENCE DATASET
# ============================================================

class SequenceDataset(Dataset):
    """
    Dataset for loading sequences of images for temporal models.

    Expected folder structure:
        root_dir/
            0/  (Not Engaged sequences)
                000000/
                    frame_0000.jpg
                    frame_0001.jpg
                    ...
                    frame_0009.jpg  (10 frames total)
                000001/
                    ...
            1/  (Engaged sequences)
                ...

    Each sample is a sequence of 10 consecutive frames.
    """
    def __init__(self, root_dir, transform=None, seq_transform=None):
        self.samples = []
        self.transform = transform          # Applied to each frame individually
        self.seq_transform = seq_transform  # Applied to entire sequence (same augmentation)

        # Load sequences from both class folders
        for label_str in ["0", "1"]:
            label_dir = os.path.join(root_dir, label_str)
            if not os.path.isdir(label_dir):
                continue

            # Each subfolder is one sequence
            for seq_id in os.listdir(label_dir):
                seq_path = os.path.join(label_dir, seq_id)

                # Collect all frame paths in this sequence
                frames = sorted([
                    os.path.join(seq_path, f)
                    for f in os.listdir(seq_path)
                    if f.endswith(".jpg")
                ])

                # Only use sequences with exactly 10 frames
                if len(frames) == SEQ_LEN:
                    self.samples.append((frames, int(label_str)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        # Use sequence-level augmentation if provided
        # This ensures all frames in sequence get the same augmentation
        if self.seq_transform:
            seq_aug = self.seq_transform
        else:
            seq_aug = None

        imgs = []
        for path in frame_paths:
            # Load image
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply sequence augmentation (same for all frames)
            if seq_aug:
                img = seq_aug(img)

            # Apply frame-level transform (normalization, etc.)
            if self.transform:
                img = self.transform(img)

            imgs.append(img)

        # Stack frames into single tensor: (SEQ_LEN, C, H, W)
        imgs = torch.stack(imgs, dim=0)
        return imgs, label



# ============================================================
# RESNET18 + GRU MODEL
# ============================================================

class ResNet18_GRU(nn.Module):
    """
    Temporal engagement detection model combining CNN and RNN.

    Architecture:
    1. ResNet18 CNN - Extracts spatial features from each frame
    2. Bidirectional GRU - Models temporal dependencies across frames
    3. FC layer - Final classification

    This model can capture temporal patterns (e.g., looking away over time)
    that single-frame models cannot detect.
    """
    def __init__(self, hidden_dim=256, num_layers=1, num_classes=2):
        super().__init__()

        # Load pretrained ResNet18 and remove final FC layer
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(base.children())[:-1])   # Output: (B, 512, 1, 1)
        self.feature_dim = 512

        # Bidirectional GRU for temporal modeling
        # Processes sequence of CNN features to capture temporal patterns
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # Process sequence forward and backward
        )

        # Classification head
        # Input size = hidden_dim * 2 because of bidirectional GRU
        self.fc = nn.Linear(hidden_dim * 2, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (B, SEQ_LEN, C, H, W)
               B = batch size
               SEQ_LEN = number of frames (10)
               C, H, W = image dimensions

        Returns:
            logits: Classification scores (B, NUM_CLASSES)
        """
        B, T, C, H, W = x.shape

        # Extract CNN features from each frame
        features = []
        for t in range(T):
            f = self.cnn(x[:, t])               # Process frame t: (B, 512, 1, 1)
            f = f.view(B, self.feature_dim)     # Flatten: (B, 512)
            features.append(f)

        # Stack features into sequence: (B, T, 512)
        sequence = torch.stack(features, dim=1)

        # Pass sequence through GRU to model temporal dependencies
        out, _ = self.gru(sequence)              # (B, T, 2*hidden_dim)

        # Use last timestep output for classification
        # This contains information from all previous frames
        last = out[:, -1, :]                     # (B, 2*hidden_dim)

        # Final classification
        logits = self.fc(last)
        return logits


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    """
    Run one epoch of training or evaluation.

    Args:
        model: Neural network model
        loader: DataLoader for sequences
        criterion: Loss function
        optimizer: Optimizer (None for evaluation mode)
        device: Device to run on (cpu/cuda)

    Returns:
        Average loss and accuracy for the epoch
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(training):
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)

            outputs = model(seqs)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

if __name__ == "__main__":
    # Create weights directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation for training sequences
    # Applied once per sequence (all frames get same augmentation)
    seq_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ])

    # Frame-level transform (applied after sequence augmentation)
    # Note: No ToPILImage here since seq_transform already converts to PIL
    frame_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1)
    ])

    # Standard normalization for Validation/test (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load datasets from folders
    print("\nLoading sequential datasets...")
    train_ds = SequenceDataset(os.path.join(ROOT_DIR, "daisee_sequential", "Train"), transform=frame_transform, seq_transform=seq_transform)
    val_ds   = SequenceDataset(os.path.join(ROOT_DIR, "daisee_sequential", "Validation"), eval_transform)
    test_ds  = SequenceDataset(os.path.join(ROOT_DIR, "daisee_sequential", "Test"), eval_transform)

    print(f"Train sequences: {len(train_ds)}")
    print(f"Val sequences:   {len(val_ds)}")
    print(f"Test sequences:  {len(test_ds)}")

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Initialize model, loss, optimizer, and scheduler
    model = ResNet18_GRU(hidden_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # Training loop with early stopping
    print("\n" + "="*60)
    print("STARTING TRAINING - ResNet18 + GRU")
    print("="*60)

    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        # Train for one epoch
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on Validation set
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)

        # Store metrics for plotting
        train_losses.append(tr_loss)
        val_accuracies.append(va_acc)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.2f}% | "
              f"Val Loss: {va_loss:.4f}, Acc: {va_acc:.2f}%")

        # Update learning rate
        scheduler.step()

        # Save best model and check early stopping
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{PATIENCE})")

            # Early stopping
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best Validation accuracy: {best_val_acc:.2f}%")
                break

    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    model.load_state_dict(torch.load(MODEL_PATH))
    _, test_acc = run_epoch(model, test_loader, criterion, None, device)

    print(f"\nFinal Results:")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print("="*60)

    # ============================================================
    # PLOT TRAINING PROGRESS
    # ============================================================

    plt.figure(figsize=(12, 5))

    # Training loss over epochs
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o')
    plt.title("Train Loss - ResNet18+GRU")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Validation accuracy over epochs
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o', color='green')
    plt.title("Validation Accuracy - ResNet18+GRU")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()