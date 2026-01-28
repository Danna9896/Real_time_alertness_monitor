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
from torchvision import transforms

# Import the three models we're comparing (resnet18, resnet18_se, ViT)
from models.resnet18 import EngagementModel as ResNetModel
from models.resnet18_se import EngagementModel as SEResNetModel
from models.ViT import load_model

# ============================================================
# PATHS
# ============================================================
# Model weights directory
MODELS_DIR = os.path.join(ROOT, "weights")

# Test dataset for visualization
DATASET_DIR = os.path.join(ROOT, "daisee_dataset", "Test")

# Trained model weights for each architecture
RESNET_WEIGHTS = os.path.join(MODELS_DIR, "Resnet18.pth")
SE_WEIGHTS     = os.path.join(MODELS_DIR, "resnet18_se.pth")
VIT_WEIGHTS    = os.path.join(MODELS_DIR, "ViT.pth")


# ============================================================
# GRAD-CAM (for CNN models)
# ============================================================
# Gradient-weighted Class Activation Mapping
# This visualizes what CNNs focus on by looking at the gradients
# flowing back to a specific layer. Higher activation = more important region.

class GradCAM:
    """
    GradCAM implementation for visualizing CNN attention.

    How it works:
    1. Forward pass: save activations from target layer
    2. Backward pass: compute gradients for predicted class
    3. Weight activations by gradients
    4. Generate heatmap showing important regions
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer  # Which layer to visualize (usually last conv layer)
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def forward_hook(m, inp, out):
            self.activations = out

        def backward_hook(m, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx):
        """Generate GradCAM heatmap for given input and class."""
        self.model.zero_grad()

        # Forward pass
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]

        # Backward pass for target class
        out[0, class_idx].backward()

        # Weight activations by gradients (global average pooling)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        # Normalize to 0-1 range
        cam = torch.relu(cam).squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ============================================================
# VIT ATTENTION ROLLOUT (for Transformer models)
# ============================================================
# Vision Transformers use self-attention to decide which image patches to focus on.
# This method combines attention weights across all layers to see the final attention map.

def vit_attention_rollout(model, x, discard_ratio=0.9):
    """
    Compute attention rollout for Vision Transformer.

    ViT splits the image into patches and uses attention to decide
    which patches are important. This function:
    1. Collects attention weights from all transformer layers
    2. Combines them by matrix multiplication (attention flow)
    3. Generates a heatmap showing which patches the model focuses on

    Args:
        model: ViT model
        x: Input image tensor
        discard_ratio: Discard low attention values (default: 0.9)

    Returns:
        Attention heatmap as numpy array
    """
    attentions = []

    # Hook to capture attention weights from each transformer layer
    def hook_fn(module, input, output):
        if output[1] is not None:
            attentions.append(output[1].detach().cpu())

    hooks = []
    original_forwards = []

    # Register hooks on attention modules
    for blk in model.encoder.layers:
        attn_module = blk.self_attention
        original_forwards.append((attn_module, attn_module.forward))

        def patched_forward(query, key, value, need_weights=True, **kwargs):
            return attn_module.__class__.forward(attn_module, query, key, value,
                                                 need_weights=True, average_attn_weights=True)

        attn_module.forward = patched_forward
        hooks.append(attn_module.register_forward_hook(hook_fn))

    # Forward pass to collect attention weights
    model.eval()
    with torch.no_grad():
        _ = model(x)

    # Clean up hooks
    for h in hooks:
        h.remove()
    for module, orig in original_forwards:
        module.forward = orig

    if not attentions:
        return np.zeros((x.shape[2], x.shape[3]))

    # Combine attention matrices across layers
    num_tokens = attentions[0].shape[-1]
    joint = torch.eye(num_tokens)

    for attn in attentions:
        attn = attn[0]
        flat = attn.flatten()

        # Discard low attention values
        _, idx = torch.topk(flat, int(len(flat) * (1 - discard_ratio)), largest=False)
        attn = attn.clone()
        attn.view(-1)[idx] = 0

        # Normalize and multiply
        attn = 0.5 * attn + 0.5 * torch.eye(num_tokens)
        attn /= attn.sum(dim=-1, keepdim=True)
        joint = attn @ joint

    # Extract attention map (skip class token)
    mask = joint[0, 1:]
    side = int(np.sqrt(mask.shape[0]))
    mask = mask.reshape(side, side).numpy()

    # Resize to image size and normalize
    mask = cv2.resize(mask, (x.shape[3], x.shape[2]))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


# ============================================================
# MAIN COMPARISON FUNCTION
# ============================================================

def compare_models(num_samples=5):
    """
    Compare attention patterns of ResNet18, SE-ResNet18, and ViT.

    This function:
    1. Loads all three trained models
    2. Randomly selects test images
    3. Generates attention heatmaps for each model
    4. Displays side-by-side comparison

    The comparison helps us understand:
    - Which parts of the face each model focuses on
    - How SE blocks change attention patterns
    - How ViT attention differs from CNN activation maps

    Args:
        num_samples: Number of test images to visualize (default: 5)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debug: Check if all required files exist
    print("\n=== Checking Files ===")
    print("ResNet weights:", RESNET_WEIGHTS, "- Exists:", os.path.exists(RESNET_WEIGHTS))
    print("SE weights:", SE_WEIGHTS, "- Exists:", os.path.exists(SE_WEIGHTS))
    print("ViT weights:", VIT_WEIGHTS, "- Exists:", os.path.exists(VIT_WEIGHTS))
    print("Test dataset:", DATASET_DIR, "- Exists:", os.path.exists(DATASET_DIR))
    print("=====================\n")

    # Load the three models
    print("Loading models...")

    # ResNet18 (baseline)
    res = ResNetModel(num_classes=2)
    res.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=device))
    res.to(device).eval()

    # ResNet18 + Squeeze-and-Excitation
    se = SEResNetModel(num_classes=2)
    se.load_state_dict(torch.load(SE_WEIGHTS, map_location=device))
    se.to(device).eval()

    # Vision Transformer
    vit = load_model(weight_path=VIT_WEIGHTS, num_classes=2)
    vit.to(device).eval()

    print("Models loaded successfully!")

    # Setup GradCAM for CNN models (visualize last convolutional layer)
    cam_res = GradCAM(res, res.model.layer4[-1].conv2)
    cam_se  = GradCAM(se, se.model.layer4[0][-1].conv2)

    # Image preprocessing transform
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
            paths += [os.path.join(folder, f) for f in os.listdir(folder)]

    # Randomly select samples
    samples = np.random.choice(paths, num_samples, replace=False)

    # Create figure with 4 columns: Original + 3 heatmaps
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4 * num_samples))

    # Column titles
    col_titles = ["Original Image", "ResNet18 Heatmap", "ResNet18+SE Heatmap", "ViT Heatmap"]
    for j in range(4):
        axes[0, j].set_title(col_titles[j], fontsize=16, fontweight='bold')

    print(f"Generating heatmaps for {num_samples} images...\n")

    print(f"Generating heatmaps for {num_samples} images...\n")

    # Process each sample image
    for i, path in enumerate(samples):
        # Load and preprocess image
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        img_np = np.array(img)

        print(f"Processing image {i+1}/{num_samples}: {os.path.basename(path)}")

        # ================================
        # 1) ResNet18 Heatmap (GradCAM)
        # ================================
        with torch.no_grad():
            pred_r = res(x).argmax(1).item()

        # Generate GradCAM heatmap
        heat_r = cv2.resize(cam_res(x, pred_r), img.size)
        heat_r = 1 - heat_r  # Invert so important areas are bright
        heat_r_color = cv2.applyColorMap((heat_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_r = cv2.addWeighted(img_np, 0.6, heat_r_color, 0.4, 0)

        # ================================
        # 2) ResNet18+SE Heatmap (GradCAM)
        # ================================
        with torch.no_grad():
            pred_s = se(x).argmax(1).item()

        # Generate GradCAM heatmap (SE model has different structure)
        heat_s = cv2.resize(cam_se(x, pred_s), img.size)
        heat_s = 1 - heat_s
        heat_s_color = cv2.applyColorMap((heat_s * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_s = cv2.addWeighted(img_np, 0.6, heat_s_color, 0.4, 0)

        # ================================
        # 3) ViT Attention Rollout
        # ================================
        # ViT uses attention mechanism instead of convolutions
        heat_v = vit_attention_rollout(vit, x)
        heat_v = 1 - heat_v
        heat_v_color = cv2.applyColorMap((heat_v * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_v = cv2.addWeighted(img_np, 0.6, heat_v_color, 0.4, 0)

        # ================================
        # Place images in figure grid
        # ================================
        images = [img_np, overlay_r, overlay_s, overlay_v]

        for j in range(4):
            axes[i, j].imshow(images[j])
            axes[i, j].axis("off")

    print("\nHeatmaps generated successfully!")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN COMPARISON
# ============================================================

if __name__ == "__main__":
    # Run comparison with 5 random test images
    # Change num_samples to visualize more or fewer images
    compare_models(num_samples=5)
