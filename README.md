00460217 – Real-Time Alertness Monitor
<div align="center"> <!-- Replace this with your own image path --> <img src="images/readme_banner.png" width="600"> <h3>Technion ECE 046211 – Deep Learning</h3> <h4>Real-Time Webcam-Based Student Engagement Detection</h4> </div>
📌 Project Overview

This project detects student engagement in real time using a webcam feed.
It includes:

Face detection (YOLOv8-Face)
<img width="1376" height="371" alt="architechture" src="https://github.com/user-attachments/assets/16f05137-5a15-40d9-b220-7986d9484c81" />

Frame-level engagement classification (ResNet-18, SE-ResNet, ViT)

Temporal modeling (GRU)

Live UI: engagement bar + focus encouragement messages

The system runs fully locally using PyTorch + OpenCV.

📁 Project Structure (short version)
Real-time_alertness_monitor/
│
├── models/                # .pth files for ResNet, SE-ResNet, ViT, GRU
├── daisee_dataset/        # Final processed DAiSEE data
├── student_concentration/ # Extracted Student Concentration dataset
│
├── preprocessing/         # Face cropping, closed-eye filtering
├── training/              # Training and evaluation scripts
├── realtime_app/          # Live webcam application (main.py)
│
├── requirements.txt
└── README.md

📥 Dataset Setup
1️⃣ DAiSEE Dataset

We provide a script that automatically downloads, extracts, crops faces, filters closed eyes, balances classes, and builds the final dataset.

Run:

python preprocessing/create_daisee_dataset.py


The output will appear in:

daisee_dataset/

2️⃣ Student Concentration Dataset (Kaggle)

▶️ Running the Real-Time App

Run:

## 🎥 

![gif_demonstration](https://github.com/user-attachments/assets/2d1948d7-1f71-4bcc-93d6-40dab6d78d2d)

python realtime_app/main.py


The app opens your webcam and displays:

Detected face bounding box

Engagement probability

Engagement bar

Encouraging messages

🧠 Models Used
Component	Purpose
YOLOv8-Face	Face detection
ResNet-18	Baseline frame classifier
SE-ResNet	Channel attention for better facial cue emphasis
ViT-B-16	Patch-based transformer classifier
GRU	Temporal smoothing over 10-frame sequences
👥 Authors


Danna Weinzinger & Saadi Saadi
Technion – Faculty of Electrical and Computer Engineering
Course: 046211 – Deep Learning

