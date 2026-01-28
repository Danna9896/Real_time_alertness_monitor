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
- Frame extraction rate  
- Target image size (224×224)  
- Closed-eye filtering threshold  
- Sequence length for temporal models (10 frames)

## Training
- Optimizer settings (AdamW)  
- Learning rate schedule  
- Batch size (images or sequences)  
- Early stopping patience  
- Number of epochs  

## Temporal Model (ResNet18+GRU)
- Sequence length (10 frames)  
- GRU hidden size  
- Frame-level embeddings from ResNet18  

## Real-Time Application
- Prediction interval (5 predictions/second)  
- Smoothing window (recent predictions)  
- Engagement bar update interval  
- Confetti / alert thresholds  
- Visual feedback color ranges  

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

### 5. Run the real-time webcam application
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


