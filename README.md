<h1 align="center">00460217 – Real-Time Alertness Monitor</h1>

<div align="center">
  <img 
       src="https://github.com/user-attachments/assets/a4809a26-f2cf-41d5-bf6c-b8db32bd2bd2"
       width="600"
  />
</div>
<p align="center">
  <a href="https://github.com/Danna9896">Danna Weinzinger</a> • 
  <a href="https://github.com/saadisaadi1">Saadi Saadi</a>
</p>

<p align="center">
  A real-time deep learning application for detecting and visualizing student engagement using webcam input.
</p>

<a href="https://github.com/user-attachments/files/24904678/Realtime_alertness_monitor_project_report.1.pdf">Project pdf</a><br>
<a href="https://www.youtube.com/watch?v=ye48xJwuano">Project presentation video</a>

## Example Screens from Our Real-Time Interface:
<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/e3a46d36-45b8-4a97-9ce3-aa7e5c7cbb9b" width="250px" />
      <br><b>App welcome screen</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8e1c8eef-ef3c-4973-b08d-f7eae6bd380e" width="250px" />
      <br><b>Unfocused example</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/d304f77a-1b09-4f02-ae0a-cb4a6144054d" width="250px" />
      <br><b>Focused example</b>
    </td>
  </tr>
</table>


# Models Implemented

1. **ResNet18 (Baseline)**  
   Single-frame CNN classifier.

2. **ResNet18 + GRU**  
   Temporal model using 10-frame sequences.

3. **ResNet18 + SE**  
   ResNet18 with Squeeze-and-Excitation channel attention.

4. **Vision Transformer (ViT)**  
   Patch-based self-attention model.

5. **Flip-Invariant ResNet18**  
   Averages predictions of original + flipped images.

6. **ResNet18 (No Augmentations)**  
   Baseline model trained without data augmentation.

# Data Preprocessing

Preprocess DAiSEE dataset (auto-downloads from Kaggle):
```bash
python preprocessing/preprocessing_daisee.py
```
Preprocess Student concentration dataset (using student dataset folder):
```bash
python preprocessing/preprocessing_sc.py
```
Create sequential dataset for GRU:
```bash
python preprocessing/preprocessing_sequential.py
```
# Parameters

## Preprocessing
- Frame extraction rate: **1 frame per second**
- Target image size: **224×224 pixels**
- Closed-eye filtering threshold: **0.5 confidence**
- Sequence length for temporal models: **10 frames**

## Training Parameters by Model

| Parameter | ResNet18 | ResNet18+SE | ResNet18+GRU | ViT | Flip-Invariant | ResNet18 (No Aug) |
|-----------|----------|-------------|--------------|-----|----------------|-------------------|
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW |
| **Learning Rate** | 5e-4 | 5e-4 | 5e-4 | 5e-5 | 5e-4 | 5e-4 |
| **Batch Size** | 16 | 16 | 8 | 16 | 16 | 16 |
| **Weight Decay** | 1e-4 | 1e-4 | 1e-4 | 0.05 | 1e-4 | 1e-4 |
| **Epochs** | 50 | 50 | 50 | 20 | 50 | 50 |
| **Input Size** | 224×224 | 224×224 | 224×224 | 224×224 | 224×224 | 224×224 |
| **Scheduler** | StepLR (step=8, γ=0.5) | StepLR (step=8, γ=0.5) | StepLR (step=8, γ=0.5) | CosineAnnealing | StepLR (step=8, γ=0.5) | StepLR (step=8, γ=0.5) |
| **Early Stopping** | 8 epochs | 8 epochs | 8 epochs | No | 8 epochs | 8 epochs |
| **Special Features** | Standard CNN | SE blocks (r=16) | GRU (h=256, l=1) | Pretrained ViT-B/16 | Horizontal flip avg | Same as ResNet18 but no augmentations |

## Temporal Model (ResNet18+GRU)
- Sequence length: **10 frames**
- GRU hidden size: **256**
- GRU layers: **1**
- Frame-level embeddings: **512-dim from ResNet18**

## Real-Time Application
- Prediction interval: **5 predictions/second (200ms)**
- Smoothing window: **Last 10 predictions**
- Engagement bar update: **Real-time**
- Alert threshold: 0% engagement bar
- Confetti threshold: 100% engagement bar
- Visual feedback: **Green (>70%), Yellow (40-70%), Red (<40%)**  

---

# Folders
- **preprocessing/** – dataset construction & filtering
- **models/** – model architectures (ResNet, SE, GRU, ViT, flip-invariant)
- **utils/** – heatmaps, confusion matrices, robustness tests
- **weights/** – trained `.pth` files
- **student_dataset/** – frame-based 2k dataset
- **sounds/** – alert audio files
- **app.py** – real-time webcam application
- **requirements.txt** – Python dependencies
- **yolov8n-face.pt** – YOLOv8 face detector

# Usage

### 1. Clone the repository
```bash
git clone https://github.com/Danna9896/Real-time_alertness_monitor.git
cd Real-time_alertness_monitor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. (Optional) Preprocess datasets — only if you want to train models
####    DAiSEE frames (ResNet / SE / ViT / Flip-Invariant)
```bash
python preprocessing/preprocessing_daisee.py
```
####    Sequential 10-frame dataset (GRU)
```bash
python preprocessing/preprocessing_sequential.py
```

### 4. (Optional) Train your own models  
You can train any of the architectures directly from the `models/` folder.  
Each script loads the dataset, trains the model, and saves the `.pth` file into `weights/`.

Examples:
```bash
python models/resnet18.py
python models/resnet18_se.py
python models/resnet18_gru.py
python models/ViT.py
python models/flip_invariant_resnet18.py
```
### 5. (Optional) Evaluate models with utility scripts

#### Generate Confusion Matrix
Evaluate any trained model and display its confusion matrix on the test set:

```bash
python utils/confusion_matrix.py
```

**To select which model to evaluate**, edit `utils/confusion_matrix.py` and change the `SELECTED_MODEL` variable (line 78):

| Index | Model | Variable to Change |
|-------|-------|-------------------|
| `0` | ResNet18 | `SELECTED_MODEL = 0` |
| `1` | ResNet18+GRU | `SELECTED_MODEL = 1` |
| `2` | ResNet18+SE | `SELECTED_MODEL = 2` |
| `3` | ViT | `SELECTED_MODEL = 3` |
| `4` | Flip-Invariant ResNet18 | `SELECTED_MODEL = 4` |
| `5` | ResNet18 (No Augmentations) | `SELECTED_MODEL = 5` |

**Example:**
```python
# In utils/confusion_matrix.py, line 78:
SELECTED_MODEL = 3  # Evaluates ViT model
```

#### Test Flip Consistency
Compare how different models handle horizontally flipped images:

```bash
python utils/flip_consistency_test.py
```

**To select which model to test**, edit `utils/flip_consistency_test.py` and change the `SELECTED_MODEL` variable (line 49):

| Index | Model | Variable to Change |
|-------|-------|-------------------|
| `0` | Standard ResNet18 | `SELECTED_MODEL = 0` |
| `1` | Flip-Invariant ResNet18 | `SELECTED_MODEL = 1` |
| `2` | ResNet18 (No Augmentations) | `SELECTED_MODEL = 2` |

**Example:**
```python
# In utils/flip_consistency_test.py, line 49:
SELECTED_MODEL = 1  # Tests Flip-Invariant ResNet18
```

This test helps verify if models are robust to horizontal flips. The Flip-Invariant model (index 1) should show the highest consistency.

### 6. Run the real-time webcam application
```bash
python app.py
```

# References

[1] Nezami, O. M., et al. *Deep Learning for Student Engagement Recognition: A Comparative Study.* ICCVW, 2019.  
[2] Gupta, A., Chakraborty, A., et al. *DAiSEE: Dataset for Affective States in E-learning Environments.* 2016.  
[3] Student Concentration Image Dataset. Kaggle. *Student Engagement Image Collection.*  
[4] Młodawski, M. *MobileNetV2 Open/Closed Eye Classifier.* 2023.  
[5] Dosovitskiy, A., et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* 2021.  
[6] Hu, J., Shen, L., Sun, G. *Squeeze-and-Excitation Networks.* CVPR, 2018.  
[7] Chung, J., Gulcehre, C., Cho, K., Bengio, Y. *Empirical Evaluation of Gated Recurrent Units on Sequence Modeling.* 2014.  

---


