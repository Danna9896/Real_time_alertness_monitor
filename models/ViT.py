# ============================================================
# IMPORTS
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

# Training parameters
LR = 5e-5                    # Lower learning rate for ViT 
BATCH_SIZE = 16
SIZE = 224                   # Input image size (224x224)
WEIGHT_DECAY = 0.05          # Higher weight decay for ViT
NUM_EPOCHS = 20            
# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "daisee_dataset")
MODEL_PATH = os.path.join(ROOT_DIR, "weights", "ViT.pth")

# Tracking metrics
train_losses = []
val_accuracies = []

# ============================================================
# MODEL LOADER
# ============================================================

def load_model(weight_path=None, num_classes=2):
    """
    Load Vision Transformer (ViT) model.

    Uses pretrained ViT-B/16 from ImageNet as backbone.
    Replaces the classification head for binary engagement detection.

    Args:
        weight_path: Path to trained weights (None for fresh model)
        num_classes: Number of output classes (default: 2)

    Returns:
        ViT model ready for training or inference
    """
    # Load pretrained ViT backbone from ImageNet
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    # Replace classifier head for our task (2 classes instead of 1000)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    # Load fine-tuned weights if provided
    if weight_path is not None:
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded model weights from: {weight_path}")

    return model


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train():
    """
    Train the ViT model on engagement dataset.

    Process:
    1. Load data with augmentation
    2. Fine-tune pretrained ViT model
    3. Track loss and Validation accuracy
    4. Save model weights and plot results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # Helps ViT ignore background noise
    ])

    # Standard normalization for Validation
    val_transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets from folder structure
    train_ds = datasets.ImageFolder(os.path.join(ROOT_DIR, "daisee_dataset", "Train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(ROOT_DIR, "daisee_dataset", "Validation"), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # Initialize model, loss, optimizer, and scheduler
    model = load_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print("\n" + "="*60)
    print("STARTING TRAINING - Vision Transformer")
    print("="*60)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0

        # Training phase
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total

        # Store metrics for plotting
        train_losses.append(running_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {running_loss:.3f} | Val Acc: {val_acc:.2f}%")

    # Save final model weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Plot training progress
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy per Epoch")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ============================================================
# TEST EVALUATION FUNCTION
# ============================================================

def evaluate_test(weight_path="weights/ViT.pth"):
    """
    Evaluate trained model on test set.

    Process:
    1. Load trained model weights
    2. Run inference on test set
    3. Calculate accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = load_model(weight_path).to(device)
    model.eval()

    # Standard preprocessing for test set
    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load test dataset
    test_ds = datasets.ImageFolder(os.path.join(ROOT_DIR, "daisee_dataset", "Test"), transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nEvaluating on {len(test_ds)} test samples...")

    # Collect predictions and ground truth
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy
    accuracy = 100 * (all_preds == all_labels).mean()
    print(f"Test Accuracy: {accuracy:.2f}%")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Train the model
    train()

    # Evaluate on test set
    evaluate_test("weights/ViT.pth")
