# ============================================================
# IMPORTS
# ============================================================
import os
import time
from collections import deque
from threading import Thread
import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO
from playsound import playsound

# Local modules
from preprocessing import preprocessing_funcs as ppf
# Model imports
from models import resnet18
from models import resnet18_se as se
from models import ViT as vit
from models import resnet18_gru as gru
from models import flip_invariant_resnet18 as fp

# =========================
# COLORS (BGR)
# =========================
BLUE       = (255,0,0)
LIGHTBLUE  = (255,255,0)
LILAC      = (203,192,255)
RED        = (0,0,255)
PINK       = (255,20,147)
GRAY       = (50,50,50)
GREEN      = (0,255,0)
ORANGERED  = (0,128,255)
YELLOW     = (0,255,255)
CYAN       = (178,255,102)
WHITE      = (255,255,255)

# Application settings
WINDOW_NAME = "Alertness Monitor"
SEQ_LEN = 10  # Number of frames for GRU temporal model

# ============================================================
# MODEL CONFIGURATIONS
# ============================================================
# Each model has:
# - name: Display name in UI
# - path: Path to trained weights
# - type: "frame" for single image input, "sequence" for temporal input
# - ctor: Constructor function to create model instance

MODELS = {
    0: {
        "name": "ResNet18",
        "path": "weights/Resnet18.pth",
        "type": "frame",
        "ctor": lambda: resnet18.EngagementModel(num_classes=2),
    },
    1: {
        "name": "ResNet18+GRU",
        "path": "weights/resnet18_gru.pth",
        "type": "sequence",  # Requires 10 consecutive frames
        "ctor": lambda: gru.ResNet18_GRU(),
    },
    2: {
        "name": "ResNet18+SE",
        "path": "weights/resnet18_se.pth",
        "type": "frame",
        "ctor": lambda: se.EngagementModel(num_classes=2),
    },
    3: {
        "name": "ViT",
        "path": "weights/ViT.pth",
        "type": "frame",
        "ctor": lambda: vit.load_model(num_classes=2),
    },
    4: {
        "name": "Flip invariant ResNet18",
        "path": "weights/resnet18_flip_invariant.pth",
        "type": "frame",
        "ctor": lambda: fp.FlipInvariantResNet18(num_classes=2),
    },
    5: {
        "name": "ResNet18 (No Augs)",
        "path": "weights/resnet18_without_augmentations.pth",
        "type": "frame",
        "ctor": lambda: resnet18.EngagementModel(num_classes=2),
    },
}


def main():
    """Main application loop for real-time alertness monitoring."""

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image preprocessing transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    # Initialize webcam and face detector
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    detector = YOLO("yolov8n-face.pt")
    cv2.namedWindow(WINDOW_NAME)

    print("\n========================================")
    print("Starting REAL-TIME Alertness Monitor...")
    print("========================================\n")

    # Timing intervals
    PREDICTION_INTERVAL = 0.2  # Run model prediction every 0.2 seconds
    AVERAGE_INTERVAL = 2.0     # Update engagement bar every 2 seconds

    # Engagement tracking
    engaged_cntr = 0
    unengaged_cntr = 0
    bar_level = 0
    BAR_SPEED = 10

    # Prediction history for averaging
    prediction_history = deque(maxlen=4)
    sequence_buffer = deque(maxlen=SEQ_LEN)  # For GRU temporal model

    # Animation state variables
    message = "Welcome to our Real-Time Alertness Monitor!"
    fade_alpha = 0.0
    intro_scale = 0.4
    SLIDE_X = -500
    last_message = message

    FADE_SPEED = 0.03
    SCALE_SPEED = 0.02
    SLIDE_SPEED = 20

    pulse_active = False
    pulse_alpha = 0
    PULSE_SPEED = 15

    confetti = []
    CONFETTI_COUNT = 25
    CONFETTI_GRAVITY = 2
    CONFETTI_FADE = 8
    confetti_triggered = False

    # Sound cooldown to prevent overlapping alerts
    last_alert_time = 0
    ALERT_COOLDOWN = 3.0  # Minimum seconds between alerts

    next_pred_time = time.time() + PREDICTION_INTERVAL
    next_avg_time = time.time() + AVERAGE_INTERVAL

    # Model selection state
    current_model_idx = 0
    model = None
    menu_visible = False
    button_rect = [0, 0, 0, 0]  # Dropdown button coordinates
    menu_rects = {}             # Menu item coordinates

    def load_model(idx: int):
        """
        Load and initialize a model by index.
        Clears prediction history and sequence buffer when switching models.
        """
        nonlocal model, sequence_buffer, prediction_history
        entry = MODELS[idx]
        name = entry["name"]
        path = entry["path"]

        print(f"[MODEL] Switching to: {name}")

        # Create model instance
        m = entry["ctor"]().to(device)

        # Load trained weights if available
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=device)
                m.load_state_dict(state)
                print(f"[MODEL] Loaded weights from {path}")
            except Exception as e:
                print(f"[MODEL] ERROR loading weights for {name} from {path}: {e}")
        else:
            print(f"[MODEL] WARNING: weight file not found: {path} (using random init)")

        m.eval()
        sequence_buffer.clear()
        prediction_history.clear()
        model = m

    load_model(current_model_idx)

    def on_mouse(event, x, y, flags, param):
        """
        Handle mouse clicks for model selection dropdown menu.
        Click button to toggle menu, click menu item to select model.
        """
        nonlocal menu_visible, current_model_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        bx1, by1, bx2, by2 = button_rect

        # Toggle menu when clicking button
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            menu_visible = not menu_visible
            return

        # Select model when clicking menu item
        if menu_visible:
            for idx, (mx1, my1, mx2, my2) in menu_rects.items():
                if mx1 <= x <= mx2 and my1 <= y <= my2:
                    if idx != current_model_idx:
                        current_model_idx = idx
                        load_model(idx)
                    menu_visible = False
                    return
            # Click outside menu closes it
            menu_visible = False

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # ============================================================
    # MAIN LOOP
    # ============================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Create canvas with top and bottom UI bars
        ui_top = 90
        ui_bottom = 70
        canvas = cv2.copyMakeBorder(
            frame,
            ui_top, ui_bottom, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        ch, cw = canvas.shape[:2]

        # ===========================
        # DYNAMIC MESSAGE DISPLAY
        # ===========================
        # Update message based on engagement counters
        if 2 <= engaged_cntr < 4:
            message = "Good job! Keep focusing"
            color = LILAC
        elif 2 <= unengaged_cntr < 4:
            message = "You seem unfocused.."
            color = LIGHTBLUE
        elif engaged_cntr >= 4:
            message = "WOW!! you're doing great!"
            color = PINK
        elif unengaged_cntr >= 4:
            message = "Maybe take a small pause?"
            color = RED
        else:
            color = BLUE

        # Reset animation when message changes
        if message != last_message:
            fade_alpha = 0.0
            intro_scale = 0.4
            SLIDE_X = -300
            last_message = message

        # Animate message entrance (fade in, scale up, slide right)
        fade_alpha = min(1.0, fade_alpha + FADE_SPEED)
        intro_scale = min(1.0, intro_scale + SCALE_SPEED)
        SLIDE_X = min(40, SLIDE_X + SLIDE_SPEED)

        draw_color = (
            int(color[0] * fade_alpha),
            int(color[1] * fade_alpha),
            int(color[2] * fade_alpha),
        )

        cv2.putText(
            canvas,
            message,
            (SLIDE_X, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            intro_scale,
            draw_color,
            2,
            cv2.LINE_AA,
        )

        # =============================
        # MODEL SELECTION DROPDOWN
        # =============================
        # Draw dropdown button in top-right corner
        btn_w, btn_h = 220, 30
        margin = 5
        bx2 = cw - margin
        bx1 = bx2 - btn_w
        by1 = margin
        by2 = by1 + btn_h
        button_rect[:] = [bx1, by1, bx2, by2]

        # Button background
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (40, 40, 40), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), WHITE, 1)

        # Display current model name
        model_name = MODELS[current_model_idx]["name"]
        cv2.putText(
            canvas,
            f"Model: {model_name}",
            (bx1 + 8, by1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            1,
            cv2.LINE_AA,
        )

        # Dropdown indicator arrow
        arrow = "v" if not menu_visible else "^"
        cv2.putText(
            canvas,
            arrow,
            (bx2 - 18, by1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            1,
            cv2.LINE_AA,
        )

        # Draw dropdown menu when visible
        menu_rects.clear()
        if menu_visible:
            entry_h = btn_h
            for i, idx in enumerate(MODELS.keys()):
                mx1 = bx1
                mx2 = bx2
                my1 = by2 + 5 + i * (entry_h + 2)
                my2 = my1 + entry_h
                menu_rects[idx] = (mx1, my1, mx2, my2)

                # Highlight selected model
                bg_color = (70, 70, 70) if idx != current_model_idx else (90, 90, 90)
                cv2.rectangle(canvas, (mx1, my1), (mx2, my2), bg_color, -1)
                cv2.rectangle(canvas, (mx1, my1), (mx2, my2), WHITE, 1)

                text = MODELS[idx]["name"]
                cv2.putText(
                    canvas,
                    text,
                    (mx1 + 8, my1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    WHITE,
                    1,
                    cv2.LINE_AA,
                )

        # =============================
        # ENGAGEMENT BAR VISUALIZATION
        # =============================
        BAR_WIDTH = 300
        BAR_HEIGHT = 18

        bar_x1 = 150
        bar_y1 = h + ui_top + 20
        bar_x2 = bar_x1 + BAR_WIDTH
        bar_y2 = bar_y1 + BAR_HEIGHT

        # Draw glow effect when bar is full
        if pulse_active:
            glow_color = GREEN
            pulse_alpha = max(0, pulse_alpha - PULSE_SPEED)
            glow_thickness = int(5 + (pulse_alpha / 255) * 10)
            cv2.rectangle(
                canvas,
                (bar_x1 - glow_thickness, bar_y1 - glow_thickness),
                (bar_x2 + glow_thickness, bar_y2 + glow_thickness),
                glow_color,
                glow_thickness,
            )
            if pulse_alpha == 0:
                pulse_active = False

        # Draw bar background
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), GRAY, -1)

        # Determine bar color based on engagement level
        if bar_level > 90:
            bar_color = GREEN
        elif bar_level > 75:
            bar_color = CYAN
        elif bar_level > 55:
            bar_color = LIGHTBLUE
        elif bar_level > 35:
            bar_color = YELLOW
        elif bar_level > 15:
            bar_color = ORANGERED
        else:
            bar_color = RED

        # Draw filled portion of bar
        fill_w = int((bar_level / 100) * BAR_WIDTH)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), bar_color, -1)

        # Display percentage
        cv2.putText(
            canvas,
            f"{int(bar_level)}%",
            (bar_x2 + 15, bar_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            bar_color,
            2,
        )

        # =============================
        # CONFETTI ANIMATION
        # =============================
        # Update and draw confetti particles
        updated = []
        for c in confetti:
            c["x"] += c["vx"]
            c["y"] += c["vy"]
            c["vy"] += CONFETTI_GRAVITY
            c["alpha"] -= CONFETTI_FADE

            if c["alpha"] <= 0:
                continue

            cv2.circle(canvas, (int(c["x"]), int(c["y"])), 4, c["color"], -1)
            updated.append(c)
        confetti = updated

        if bar_level < 90:
            confetti_triggered = False

        # =============================
        # DISPLAY FRAME
        # =============================
        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # =============================
        # MODEL PREDICTION
        # =============================
        now = time.time()

        if now >= next_pred_time:
            cropped, found = ppf.detect_and_crop_face(detector, frame, (224, 224))

            if found and model is not None:
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                face_tensor = transform(rgb)

                mtype = MODELS[current_model_idx]["type"]

                if mtype == "sequence":
                    # GRU model: requires 10 consecutive frames
                    sequence_buffer.append(face_tensor)
                    if len(sequence_buffer) == SEQ_LEN:
                        seq_tensor = torch.stack(list(sequence_buffer), dim=0)
                        seq_tensor = seq_tensor.unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = model(seq_tensor)
                            probs = torch.softmax(output, dim=1)
                            pred = torch.argmax(probs, 1).item()
                        prediction_history.append(pred)
                else:
                    # Frame-based model: single image input
                    input_tensor = face_tensor.unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred = torch.argmax(probs, 1).item()
                    prediction_history.append(pred)
            else:
                # No face detected - clear prediction history to stop engagement updates
                prediction_history.clear()

            next_pred_time = now + PREDICTION_INTERVAL

        # =============================
        # UPDATE ENGAGEMENT BAR
        # =============================
        # Only update if we have predictions (means face was recently detected)
        if now >= next_avg_time and prediction_history:
            engaged = prediction_history.count(1)
            unengaged = prediction_history.count(0)
            prev_bar_level = bar_level

            # Update bar level based on majority prediction
            if engaged > unengaged:
                engaged_cntr += 1
                unengaged_cntr = 0
                bar_level += BAR_SPEED
            else:
                unengaged_cntr += 1
                engaged_cntr = 0
                bar_level -= BAR_SPEED

            bar_level = max(0, min(bar_level, 100))

            # Play alert sound when engagement drops to zero
            if bar_level == 0 and (now - last_alert_time) > ALERT_COOLDOWN:
                # Play sound in separate thread
                Thread(target=lambda: playsound("sounds/alert0bar.mp3"), daemon=True).start()
                last_alert_time = now

            # Trigger confetti when reaching 100%
            if prev_bar_level < 100 and bar_level >= 100 and not confetti_triggered:
                confetti_triggered = True
                for _ in range(CONFETTI_COUNT):
                    confetti.append({
                        "x": bar_x1 + BAR_WIDTH // 2 + int((torch.rand(1).item() - 0.5) * 120),
                        "y": bar_y1 - 20,
                        "vx": int((torch.rand(1).item() - 0.5) * 8),
                        "vy": -int(torch.rand(1).item() * 15),
                        "color": (
                            int(torch.randint(0, 255, (1,)).item()),
                            int(torch.randint(0, 255, (1,)).item()),
                            int(torch.randint(0, 255, (1,)).item()),
                        ),
                        "alpha": 255,
                    })

            # Activate glow effect when bar is full
            if bar_level == 100 and not pulse_active:
                pulse_active = True
                pulse_alpha = 255

            next_avg_time = now + AVERAGE_INTERVAL

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nApplication closed.\n")


if __name__ == "__main__":
    main()
