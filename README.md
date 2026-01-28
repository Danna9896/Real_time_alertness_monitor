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
<a href="https://www.youtube.com/watch?v=ye48xJwuano">Project Presentation</a>

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

- **models/** – trained weights (.pth)  
- **preprocessing/** – scripts for dataset creation and filtering  
- **sequences/** – scripts for GRU sequence generation  
- **assets/** – images used in README/UI  
- **app.py** – main real-time webcam application  
- **train_resnet.py** – ResNet training script  
- **train_gru.py** – GRU sequence model training  

---

# Reference

[1] Nezami, O. M., et al. *Deep Learning for Student Engagement Recognition.* ICCVW 2019.  
[2] Hu, J., Shen, L., Sun, G. *Squeeze-and-Excitation Networks.* CVPR 2018.  
[3] Dosovitskiy, A., et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition.* 2021.  
[4] Chung, J., Gulcehre, C., Cho, K., Bengio, Y. *Gated Recurrent Units for Sequence Modeling.* 2014.  

---

# Project Video Demonstration
https://www.youtube.com/watch?v=ye48xJwuano
