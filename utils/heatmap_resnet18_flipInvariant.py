# ============================================================
# IMPORTS
# ============================================================
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Import the two models we're comparing (standard resnet18 and flip-invariant resnet18)
from models.resnet18 import EngagementModel as ResNetModel
from models.flip_invariant_resnet18 import FlipInvariantResNet18

# ============================================================
# PATHS
# ============================================================
# Model weights directory
MODELS_DIR = os.path.join(ROOT, "weights")

# Test dataset for visualization
DATASET_DIR = os.path.join(ROOT, "daisee_dataset", "Test")

# Trained model weights
RESNET_WEIGHTS = os.path.join(MODELS_DIR, "Resnet18.pth")
FLIP_INV_WEIGHTS = os.path.join(MODELS_DIR, "resnet18_flip_invariant.pth")


# ============================================================
# GRAD-CAM FOR STANDARD RESNET18
# ============================================================
# Gradient-weighted Class Activation Mapping
# Shows which parts of the image the CNN focuses on

class GradCAM:
    """
    GradCAM for standard ResNet18.

    How it works:
    1. Forward pass: save activations from the last conv layer
    2. Backward pass: compute gradients for predicted class
    3. Weight activations by gradients to see what's important
    4. Generate heatmap showing which regions matter most
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Hook into forward and backward pass to grab activations and gradients."""
        def forward_hook(m, inp, out):
            self.activations = out.detach()

        def backward_hook(m, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x):
        """Generate GradCAM heatmap for the input."""
        self.model.zero_grad()

        # Forward pass
        out = self.model(x)
        pred = out.argmax(1).item()

        # Backward pass for predicted class
        one_hot = torch.zeros_like(out)
        one_hot[0, pred] = 1
        out.backward(gradient=one_hot, retain_graph=True)

        # Compute weighted activation map
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Weight each channel by its average gradient
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        cam = (weights * activations).sum(dim=0)  # (H, W)

        # ReLU + normalize to 0-1
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), pred


# ============================================================
# GRAD-CAM FOR FLIP-INVARIANT RESNET18
# ============================================================
# The flip-invariant model processes both original and flipped images
# and averages their predictions. We need to combine gradients from
# both branches to see what the full model focuses on.

class FlipInvariantGradCAM:
    """
    GradCAM for Flip-Invariant ResNet18.

    This model runs TWO forward passes (original + flipped image)
    and averages the predictions. To see what it actually focuses on,
    we compute gradients through BOTH branches and combine them.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations_stack = []  # Stack for forward pass activations
        self.paired = []  # List of (activation, gradient) tuples
        self._register_hooks()

    def _register_hooks(self):
        """Hook into forward and backward passes."""
        def forward_hook(m, inp, out):
            # Push activation onto stack (forward order)
            self.activations_stack.append(out.detach())

        def backward_hook(m, grad_in, grad_out):
            # Pop activation from stack (backward order = reverse)
            # This correctly pairs each gradient with its activation
            if len(self.activations_stack) > 0:
                act = self.activations_stack.pop()
                grad = grad_out[0].detach()
                self.paired.append((act, grad))

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x):
        """Generate combined GradCAM heatmap from both branches."""
        self.model.eval()

        # Clear previous state
        self.activations_stack = []
        self.paired = []

        # Forward pass through flip-invariant model
        # This triggers forward hooks TWICE (original + flipped branches)
        out = self.model(x)
        pred = out.argmax(1).item()

        # Backward pass computes gradients through BOTH branches
        self.model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, pred] = 1
        out.backward(gradient=one_hot, retain_graph=True)

        # Now self.paired has 2 tuples: [(flip_act, flip_grad), (orig_act, orig_grad)]
        # They're in reverse order because backward hooks run backwards
        if len(self.paired) != 2:
            raise RuntimeError(f"Expected 2 activation-gradient pairs, got {len(self.paired)}")

        # Extract pairs (remember: reversed order)
        flip_act, flip_grad = self.paired[0][0][0], self.paired[0][1][0]  # Flipped branch
        orig_act, orig_grad = self.paired[1][0][0], self.paired[1][1][0]  # Original branch

        # Flip the flipped branch back to original orientation
        flip_act_corrected = torch.flip(flip_act, dims=[2])  # Flip along width
        flip_grad_corrected = torch.flip(flip_grad, dims=[2])

        # Compute GradCAM for each branch
        weights_orig = orig_grad.mean(dim=(1, 2), keepdim=True)
        weights_flip = flip_grad_corrected.mean(dim=(1, 2), keepdim=True)

        cam_orig = (weights_orig * orig_act).sum(dim=0)
        cam_flip = (weights_flip * flip_act_corrected).sum(dim=0)

        # Average the two branches (since final prediction is averaged)
        cam = (cam_orig + cam_flip) / 2.0

        # ReLU + normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), pred


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def apply_heatmap_overlay(img_np, heatmap, alpha=0.5):
    """
    Overlay heatmap on original image.

    Args:
        img_np: Original image (H, W, 3) in [0, 255]
        heatmap: Heatmap (H, W) in [0, 1]
        alpha: Transparency (0.5 = 50% overlay)

    Returns:
        Overlayed image (H, W, 3)
    """
    # Resize heatmap to match image size
    h, w = img_np.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))

    # Convert heatmap to color (JET colormap: blue=low, red=high)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay = (heatmap_color * alpha + img_np * (1 - alpha)).astype(np.uint8)
    return overlay


# ============================================================
# MAIN COMPARISON FUNCTION
# ============================================================

def compare_resnet_and_flip_invariant(num_samples=5):
    """
    Compare standard ResNet18 vs Flip-Invariant ResNet18.

    This visualization shows:
    - How standard ResNet18 can give different predictions for flipped images
    - How flip-invariant ResNet18 maintains consistent predictions
    - What regions each model focuses on

    Layout for each image:
    Row 1: Original image | ResNet18 heatmap | Flip-Inv heatmap
    Row 2: Flipped image  | ResNet18 heatmap | Flip-Inv heatmap

    Args:
        num_samples: Number of test images to visualize
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if files exist
    print("\n=== Checking Files ===")
    print("ResNet weights:", RESNET_WEIGHTS, "- Exists:", os.path.exists(RESNET_WEIGHTS))
    print("Flip-Inv weights:", FLIP_INV_WEIGHTS, "- Exists:", os.path.exists(FLIP_INV_WEIGHTS))
    print("Test dataset:", DATASET_DIR, "- Exists:", os.path.exists(DATASET_DIR))
    print("=====================\n")

    # Load models
    print("Loading models...")

    # Standard ResNet18
    resnet = ResNetModel(num_classes=2)
    resnet.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=device))
    resnet.to(device).eval()

    # Flip-Invariant ResNet18
    flip_inv = FlipInvariantResNet18(num_classes=2)
    flip_inv.load_state_dict(torch.load(FLIP_INV_WEIGHTS, map_location=device))
    flip_inv.to(device).eval()

    print("Models loaded successfully!")

    # Setup GradCAM for both models
    cam_resnet = GradCAM(resnet, resnet.model.layer4[-1])
    cam_flip_inv = FlipInvariantGradCAM(flip_inv, flip_inv.backbone.layer4[-1])

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Collect test images from both classes
    print("Loading test images...")
    paths = []
    for cls in ["0", "1"]:
        folder = os.path.join(DATASET_DIR, cls)
        if os.path.exists(folder):
            paths += [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if len(paths) == 0:
        print("ERROR: No test images found!")
        print("Make sure you've run setup_sample_data.py first.")
        return

    # Randomly select samples
    samples = np.random.choice(paths, min(num_samples, len(paths)), replace=False)

    class_names = ["Not Engaged", "Engaged"]

    print(f"\nGenerating heatmaps for {len(samples)} images...\n")

    # Process each image
    for img_idx, path in enumerate(samples):
        # Load and preprocess image
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        img_np = np.array(img)

        # Create flipped version
        x_flipped = torch.flip(x, dims=[3])
        img_flipped_np = np.fliplr(img_np)  # Flip image left-right using numpy

        print(f"Processing image {img_idx+1}/{len(samples)}: {os.path.basename(path)}")

        # ================================
        # Generate heatmaps for ORIGINAL image
        # ================================

        # ResNet18 on original image
        heat_resnet_orig, pred_resnet_orig = cam_resnet(x.clone())

        # Flip-Invariant on original image
        heat_flip_orig, pred_flip_orig = cam_flip_inv(x.clone())

        # ================================
        # Generate heatmaps for FLIPPED image
        # ================================

        # ResNet18 on flipped image
        heat_resnet_flip, pred_resnet_flip = cam_resnet(x_flipped.clone())

        # Flip-Invariant on flipped image
        heat_flip_flip, pred_flip_flip = cam_flip_inv(x_flipped.clone())

        # ================================
        # Create overlays
        # ================================

        overlay_resnet_orig = apply_heatmap_overlay(img_np, heat_resnet_orig, alpha=0.5)
        overlay_flip_orig = apply_heatmap_overlay(img_np, heat_flip_orig, alpha=0.5)

        overlay_resnet_flip = apply_heatmap_overlay(img_flipped_np, heat_resnet_flip, alpha=0.5)
        overlay_flip_flip = apply_heatmap_overlay(img_flipped_np, heat_flip_flip, alpha=0.5)

        # ================================
        # Create visualization
        # Layout: 2 rows x 3 columns
        # Row 1: Original image, ResNet18 heatmap, Flip-Inv heatmap
        # Row 2: Flipped image, ResNet18 heatmap, Flip-Inv heatmap
        # ================================

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(overlay_resnet_orig)
        axes[0, 1].set_title(f"ResNet18\n{class_names[pred_resnet_orig]}", fontsize=11)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(overlay_flip_orig)
        axes[0, 2].set_title(f"Flip-Invariant\n{class_names[pred_flip_orig]}", fontsize=11)
        axes[0, 2].axis('off')

        # Row 2: Flipped image
        axes[1, 0].imshow(img_flipped_np)
        axes[1, 0].set_title("Flipped Image", fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(overlay_resnet_flip)
        axes[1, 1].set_title(f"ResNet18\n{class_names[pred_resnet_flip]}", fontsize=11)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(overlay_flip_flip)
        axes[1, 2].set_title(f"Flip-Invariant\n{class_names[pred_flip_flip]}", fontsize=11)
        axes[1, 2].axis('off')

        # Overall title
        fig.suptitle(
            f"ResNet18 vs Flip-Invariant ResNet18 - Image {img_idx+1}",
            fontsize=14, fontweight='bold'
        )

        plt.tight_layout()
        plt.show()

    print("\nAll heatmaps displayed!")


# ============================================================
# RUN COMPARISON
# ============================================================

if __name__ == "__main__":
    # Run comparison with 5 random test images
    # You can change this number to visualize more or fewer images
    compare_resnet_and_flip_invariant(num_samples=5)

