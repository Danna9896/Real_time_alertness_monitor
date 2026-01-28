# ============================================================
# IMPORTS
# ============================================================
import cv2

#params
SIZE = 224

def detect_and_crop_face(detector, frame_bgr, target_size=(SIZE, SIZE)):
    """
    Detect face in frame using YOLOv8, crop and resize it.

    Args:
        detector: YOLO face detector (ultralytics.YOLO)
        frame_bgr: Input frame in BGR format
        target_size: Target size for resizing (width, height)

    Returns:
        Tuple of (cropped_resized_face, found_flag)
    """
    # Run YOLO inference with lower confidence threshold for better detection
    # conf=0.3 means detect faces with 30%+ confidence
    results = detector(frame_bgr, verbose=False, conf=0.3)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return None, False

    # Choose the detection with highest confidence
    boxes = results.boxes
    scores = boxes.conf.cpu().numpy()

    # Filter for minimum confidence of 0.4 to avoid false positives
    valid_mask = scores >= 0.4
    if not valid_mask.any():
        return None, False

    # Get best detection among valid ones
    valid_scores = scores[valid_mask]
    valid_boxes = boxes.xyxy[valid_mask]
    best_idx = valid_scores.argmax()

    # Get bounding box (x1, y1, x2, y2)
    x1, y1, x2, y2 = valid_boxes[best_idx].cpu().numpy().astype(int)

    # Crop and resize with padding
    resized_face = crop_and_resize_face(
        frame_bgr, x1, y1, x2, y2, target_size=target_size
    )

    return resized_face, True


def crop_and_resize_face(frame_bgr, x1, y1, x2, y2, target_size=(SIZE, SIZE)):
    """
    Crop a face region from frame and resize to target size for training.

    Args:
        frame_bgr: Input image in BGR format
        x1, y1, x2, y2: Bounding box coordinates
        target_size: Tuple (width, height) for resizing, default (224, 224)

    Returns:
        Resized cropped face image
    """
    h, w = frame_bgr.shape[:2]

    # Add 15% padding around face for better context
    face_w = x2 - x1
    face_h = y2 - y1
    pad_w = int(face_w * 0.15)
    pad_h = int(face_h * 0.15)

    # Apply padding and clip to frame boundaries
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    # Crop the face region
    cropped_face = frame_bgr[y1:y2, x1:x2]

    # Resize to target size
    resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_LINEAR)

    return resized_face
