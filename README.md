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


## What the System Does

<p align="center">
  <img src="assets/blue_pipeline.png" width="85%">
</p>

A webcam stream is processed in real time:

1. **YOLOv8-face** detects and crops the face  
2. A deep-learning model classifies **Engaged vs Not Engaged**  
3. A smoothing layer stabilizes noisy frame-level predictions  
4. A **blue-themed engagement bar** displays:
   - Supportive messages  
   - Glow animations  
   - Confetti at 100%  
   - Alert sound at 0%

---

## Key Features (Blue Theme)

### 🎯 Real-time engagement detection
- 5 inferences per second  
- Majority-vote smoothing  
- Stable even during blinks and micro-expressions  

### 🔵 Clean blue visual design
- Soft gradients  
- Rounded overlays  
- Readable minimal UI  

### 📊 Multiple deep-learning backbones tested
- ResNet18 (baseline)  
- ResNet18 + SE  
- Vision Transformer (ViT)  
- **ResNet18 + GRU (best performance)**  

### 🧪 Dataset preprocessing & cleaning
- YOLO-based face cropping  
- Open/closed-eye filtering  
- Balanced binary labels  
- 10-frame temporal sequence generation  

---

## Example Results

<p align="center">
  <img src="assets/blue_heatmaps.png" width="90%">
</p>

- **ViT** → noisy and diffuse attention  
- **ResNet18** → stable spatial focus  
- **SE-ResNet18** → tighter eye/mouth emphasis  
- **GRU model** → best temporal consistency  

---

## Quick Start

### Install
```bash
git clone https://github.com/yourusername/Real-time-alertness-monitor.git
cd Real-time-alertness-monitor
pip install -r requirements.txt



## References

Methods & previous research that guided our system:

- **Nezami et al. (2019)** – Engagement recognition from facial expressions  
- **Squeeze-and-Excitation Networks (Hu et al., 2018)**  
- **Vision Transformer (Dosovitskiy et al., 2021)**  
- **Temporal sequence modeling with GRUs (Chung et al., 2014)**  
