import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# ============================================================
# Load GRU Model
# ============================================================

class ResNet18_GRU(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1, num_classes=2):
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = 512

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape

        features = []
        for t in range(T):
            f = self.cnn(x[:, t]).view(B, self.feature_dim)
            features.append(f)

        seq = torch.stack(features, dim=1)
        out, _ = self.gru(seq)
        logits = self.fc(out[:, -1])
        return logits, out


class FrameResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


# ============================================================
# Setup
# ============================================================

# Get parent directory (root of project)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

gru_model = ResNet18_GRU().to(DEVICE)
gru_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "weights", "resnet18_gru.pth"), map_location=DEVICE))
gru_model.eval()

frame_model = FrameResNet().to(DEVICE)
frame_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "weights", "Resnet18.pth"), map_location=DEVICE))
frame_model.eval()

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

SEQ_LEN = 10
TEST_PATH = os.path.join(ROOT_DIR, "daisee_sequential", "Test")


# ============================================================
# Load sequence
# ============================================================

def load_random_sequence():
    cls = np.random.choice(["0", "1"])
    seq_name = np.random.choice(os.listdir(os.path.join(TEST_PATH, cls)))
    folder = os.path.join(TEST_PATH, cls, seq_name)

    frames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")])

    imgs = []
    for path in frames:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(eval_transform(img))

    stack = torch.stack(imgs).unsqueeze(0).to(DEVICE)
    return stack, int(cls)


# ============================================================
# Compute real predictions
# ============================================================

@torch.no_grad()
def compute_curves(seq):

    # ------------------ CNN frame-wise predictions ------------------
    frame_only = []
    for t in range(SEQ_LEN):
        logits = frame_model(seq[:, t])
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        frame_only.append(prob)

    # ------------------ GRU incremental predictions ------------------
    gru_per_frame = []
    for t in range(SEQ_LEN):
        partial_seq = seq[:, :t+1]
        logits_t, _ = gru_model(partial_seq)
        prob_t = torch.softmax(logits_t, dim=1)[0, 1].item()
        gru_per_frame.append(prob_t)

    # ------------------ GRU final ------------------
    final_logits, _ = gru_model(seq)

    return frame_only, gru_per_frame


# ============================================================
# Plot
# ============================================================

LEFT_COLS = 5
RIGHT_COLS = 5

seq, label = load_random_sequence()
cnn_curve, gru_curve = compute_curves(seq)

gt_y = float(label)
gt_label_name = "Engaged" if label == 1 else "Not Engaged"

# Restore images for display
frames = []
for t in range(SEQ_LEN):
    img = seq[0, t].permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    frames.append(img)


fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, LEFT_COLS + RIGHT_COLS)

# ----------- montage -----------
for i in range(SEQ_LEN):
    row = i // 5
    col = i % 5
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(frames[i])
    ax.axis("off")
    ax.set_title(f"{i+1}", fontsize=8)

# ----------- curve ------------
ax_curve = fig.add_subplot(gs[:, LEFT_COLS:])

x = np.arange(1, SEQ_LEN + 1)

ax_curve.plot(x, cnn_curve, marker='o', label="ResNet frame-wise", linewidth=2)
ax_curve.plot(x, gru_curve, marker='o', label="GRU incremental", linewidth=2)
ax_curve.axhline(gt_y, color='green', linestyle='--',
                 label=f"Ground Truth = {gt_label_name}")

ax_curve.set_title("Temporal Stability Curve", fontsize=14)
ax_curve.set_xlabel("Frame index")
ax_curve.set_ylabel("Engaged Probability")
ax_curve.grid(True)
ax_curve.set_ylim(0, 1)
ax_curve.legend()

plt.tight_layout()
plt.show()


