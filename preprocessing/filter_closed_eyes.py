# ============================================================
# IMPORTS
# ============================================================
import os
import shutil
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm


def filter_closed_eyes(input_root, output_root, confidence_threshold=0.6):
    """
    Filter out images with closed eyes from the engaged dataset.

    Args:
        input_root: Path to input folder with 0/ and 1/ subfolders
        output_root: Path to output folder for filtered images
        confidence_threshold: Confidence level to consider eyes as closed (default: 0.6)
    """

    # Create output directories
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, "removed"), exist_ok=True)

    # Setup device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained eye classification model from HuggingFace
    # This model can detect whether eyes are open or closed
    processor = AutoImageProcessor.from_pretrained(
        "MichalMlodawski/open-closed-eye-classification-mobilev2"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "MichalMlodawski/open-closed-eye-classification-mobilev2"
    ).to(device)
    model.eval()

    # Track statistics
    kept_total = {0: 0, 1: 0}
    removed_total = 0

    # Process each class folder (0 and 1)
    for label in ["0", "1"]:
        src_class_dir = os.path.join(input_root, label)
        dst_class_dir = os.path.join(output_root, label)
        os.makedirs(dst_class_dir, exist_ok=True)

        if not os.path.exists(src_class_dir):
            continue

        files = os.listdir(src_class_dir)
        print(f"\nFiltering closed eyes for class {label}: {len(files)} images")

        for fname in tqdm(files, desc=f"Filtering class {label}", ncols=80):
            src_path = os.path.join(src_class_dir, fname)

            # Load image
            try:
                img = Image.open(src_path).convert("RGB")
            except:
                continue

            # Preprocess image for model
            inputs = processor(images=img, return_tensors="pt").to(device)

            # Run prediction
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                pred_idx = logits.argmax(-1).item()
                conf = probs[0][pred_idx].item()

            # Check if eyes are closed with high confidence
            is_closed = (pred_idx == 0 and conf >= confidence_threshold)
            engaged = (label == "1")

            # Only remove images with closed eyes from engaged class
            # Closed eyes in "not engaged" class might be relevant
            if is_closed and engaged:
                removed_total += 1
                removed_name = f"closed_{removed_total:06d}.jpg"
                shutil.copy2(src_path, os.path.join(output_root, "removed", removed_name))
            else:
                # Keep the image
                kept_total[int(label)] += 1
                dst = os.path.join(dst_class_dir, f"{label}_{kept_total[int(label)]:06d}.jpg")
                shutil.copy2(src_path, dst)

    # Print summary
    print("\n=== Closed-eye filtering complete ===")
    print(f"Kept: class 0 → {kept_total[0]}, class 1 → {kept_total[1]}")
    print(f"Removed closed eyes: {removed_total}")
    print(f"Saved to: {output_root}")