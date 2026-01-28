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


# Real-time Alertness Monitor

Python implementation of a real-time webcam system that detects **student engagement** using deep learning  
(ResNet18, SE-ResNet18, ViT, and ResNet18+GRU), combined with a clean **blue-themed UI**.

<p align="center">
  <img src="assets/preview_app_1.png" width="45%">
  <img src="assets/preview_app_2.png" width="45%">
</p>

---

## Based on methods from:

- *Nezami et al. (2019)* — Student Engagement Recognition  
- *Squeeze-and-Excitation Networks (Hu et al., 2018)* — Channel attention  
- *Vision Transformer (Dosovitskiy et al., 2021)* — Patch-based attention  
- *GRU sequence modeling (Chung et al., 2014)* — Temporal reasoning  

These inspired the model architectures and analysis used in this project.

Project PDF:  
**Realtime Alertness Monitor – Report**  
(Place your PDF link here)

---

In this repository, we explain and implement our real-time alertness system.  
This is an overview of what we aim to achieve:

<p align="center">
  <img src="assets/blue_overview_1.png" width="23%">
  <img src="assets/blue_overview_2.png" width="23%">
  <img src="assets/blue_overview_3.png" width="23%">
  <img src="assets/blue_overview_4.png" width="23%">
</p>

---

We divide the workflow into 3 main steps:

1. **Face detection & preprocessing**  
   Cropping faces from webcam frames with YOLOv8-face, cleaning closed-eye images, balancing dataset.

2. **Engagement prediction**  
   Multiple deep-learning models (ResNet18 baseline, SE block, ViT, GRU temporal model).

3. **Real-time UI**  
   Smooth predictions, engagement bar, glow animation, supportive messages, sound alerts.

Combining these yields the final real-time system.  
The workflow can be depicted as follows:

<p align="center">
  <img src="assets/blue_pipeline.png" width="90%">
</p>

---

### Another example:

<p align="center">
  <img src="assets/blue_example_1.png" width="45%">
  <img src="assets/blue_example_2.png" width="45%">
</p>

---

# Usage

```python
from models.resnet import load_resnet18
from realtime.app import run_app

# Load trained model
model = load_resnet18("models/Resnet18_cleaned.pth")

# Run webcam application
run_app(model)

