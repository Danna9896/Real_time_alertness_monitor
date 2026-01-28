# ============================================================
# IMPORTS
# ============================================================
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

#local imports
from models.resnet18 import EngagementModel
from models.flip_invariant_resnet18 import FlipInvariantResNet18


# ============================================================
# PARAMETERS
# ============================================================
MODEL_PATH = os.path.join(ROOT_DIR, "weights", "Resnet18.pth")
TEST_DIR = os.path.join(ROOT_DIR, "daisee_dataset", "Test")


# ============================================================
# MODEL SELECTION DICTIONARY (like confusion_matrix.py)
# ============================================================
MODELS = {
    0: {
        "name": "Standard ResNet18",
        "class": EngagementModel,
        "weights": os.path.join(ROOT_DIR, "weights", "Resnet18.pth"),
    },
    1: {
        "name": "Flip-Invariant ResNet18",
        "class": FlipInvariantResNet18,
        "weights": os.path.join(ROOT_DIR, "weights", "resnet18_flip_invariant.pth"),
    },
    2: {
        "name": "ResNet18 (No Augmentations)",
        "class": EngagementModel,
        "weights": os.path.join(ROOT_DIR, "weights", "resnet18_without_augmentations.pth"),
    },
}

# Choose which model to test: 0 (Standard ResNet18), 1 (Flip-Invariant ResNet18), 2 (ResNet18 No Augmentations)
SELECTED_MODEL = 2


# ============================================================
# DATASET
# ============================================================

class SimpleFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label in ["0", "1"]:
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for fname in os.listdir(label_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((
                        os.path.join(label_dir, fname),
                        int(label)
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label, image_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model config
    model_cfg = MODELS[SELECTED_MODEL]
    print(f"Loading model: {model_cfg['name']}")
    print(f"Weights: {model_cfg['weights']}")

    # Load model
    model = model_cfg["class"]()
    state_dict = torch.load(model_cfg["weights"], map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = SimpleFolderDataset(TEST_DIR, transform=transform)

    print(f"Loaded {len(dataset)} test images")

    # flip consistency
    total = len(dataset)
    num_diff = 0

    for idx in tqdm(range(total), desc="Testing", ncols=80):
        image, label, path = dataset[idx]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_orig = int(model(image).argmax(dim=1).item())
            pred_flip = int(model(torch.flip(image, dims=[3])).argmax(dim=1).item())

        if pred_orig != pred_flip:
            num_diff += 1

    print(f"\nTotal images: {total}")
    print(f"Different after flip: {num_diff}")
    print(f"Consistency: {100.0 * (total - num_diff) / total:.2f}%")

    # Create plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        ["Consistent", "Different"],
        [total - num_diff, num_diff],
        color=["green", "red"],
        alpha=0.7,
        edgecolor='black'
    )
    plt.ylabel("Count", fontsize=12)
    plt.title(f"Flip Consistency Test\nConsistency: {100.0 * (total - num_diff) / total:.2f}%",
              fontsize=14, fontweight='bold')

    # Add count labels on top of bars
    for bar, count in zip(bars, [total - num_diff, num_diff]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.show()
