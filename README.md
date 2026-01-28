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
Preprocess Student concentration dataset:
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
git clone https://github.com/<your-username>/Real-time_alertness_monitor.git
cd Real-time_alertness_monitor

### 2. Install dependencies
pip install -r requirements.txt

### 3. (Optional) Preprocess datasets — only if you want to train models
####    DAiSEE frames (ResNet / SE / ViT / Flip-Invariant)
python preprocessing/preprocessing_daisee.py

####    Sequential 10-frame dataset (GRU)
python preprocessing/preprocessing_sequential.py

### 4. Run the real-time webcam application
python app.py

# References

[1] Nezami, O. M., et al. *Deep Learning for Student Engagement Recognition: A Comparative Study.* ICCVW, 2019.  
[2] Gupta, A., Chakraborty, A., et al. *DAiSEE: Dataset for Affective States in E-learning Environments.* 2016.  
[3] Student Concentration Image Dataset. Kaggle. *Student Engagement Image Collection.*  
[4] Młodawski, M. *MobileNetV2 Open/Closed Eye Classifier.* 2023.  
[5] Dosovitskiy, A., et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* 2021.  
[6] Hu, J., Shen, L., Sun, G. *Squeeze-and-Excitation Networks.* CVPR, 2018.  
[7] Chung, J., Gulcehre, C., Cho, K., Bengio, Y. *Empirical Evaluation of Gated Recurrent Units on Sequence Modeling.* 2014.  

---


