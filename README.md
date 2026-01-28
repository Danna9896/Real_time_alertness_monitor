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

# Parameters

## Preprocessing
- **FRAME_INTERVAL**: sampling rate for DAiSEE frames  
- **TARGET_SIZE**: final face crop size (224×224)  
- **OPEN_EYE_THRESHOLD**: confidence threshold for filtering  

## Training
- **learning_rate**: optimizer learning rate  
- **weight_decay**: L2 regularization  
- **batch_size**: batch size for training  
- **seq_len**: number of frames used by GRU sequence model  
- **hidden_size**: GRU hidden units  

## Real-time app
- **PREDICTION_INTERVAL**: model prediction frequency  
- **VOTE_WINDOW**: number of predictions used for smoothing  
- **ALERT_THRESHOLDS**: values controlling confetti or alerts  

---

# Folders
- **preprocessing/** – dataset construction & filtering
- **models/** – model architectures (ResNet, SE, GRU, ViT, flip-invariant)
- **utils/** – heatmaps, confusion matrices, robustness tests
- **weights/** – trained `.pth` files
- **daisee_dataset/** – frame-based dataset
- **daisee_sequential/** – sequential dataset for GRU
- **sounds/** – alert audio files
- **app.py** – real-time webcam application
- **config.py** – centralized configuration
- **requirements.txt** – Python dependencies
- **yolov8n-face.pt** – YOLOv8 face detector

  
# Reference

# References

[1] Nezami, O. M., et al. *Deep Learning for Student Engagement Recognition: A Comparative Study.* ICCVW, 2019.  
[2] Gupta, A., Chakraborty, A., et al. *DAiSEE: Dataset for Affective States in E-learning Environments.* 2016.  
[3] Student Concentration Image Dataset. Kaggle. *Student Engagement Image Collection.*  
[4] Młodawski, M. *MobileNetV2 Open/Closed Eye Classifier.* 2023.  
[5] Dosovitskiy, A., et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* 2021.  
[6] Hu, J., Shen, L., Sun, G. *Squeeze-and-Excitation Networks.* CVPR, 2018.  
[7] Chung, J., Gulcehre, C., Cho, K., Bengio, Y. *Empirical Evaluation of Gated Recurrent Units on Sequence Modeling.* 2014.  

---


