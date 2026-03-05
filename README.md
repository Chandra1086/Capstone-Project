# 🚦 Automated Hand Signal and Traffic Signal Identification for Autonomous Vehicles

## 📌 Project Overview

This project presents an **AI-based computer vision system** capable of detecting **traffic light signals and traffic police hand gestures** in real time. The system uses **deep learning and convolutional neural networks (CNN)** to analyze visual input from a webcam and classify traffic instructions.

The project integrates **real-time signal detection with a virtual autonomous vehicle simulation**, where detected signals directly influence the vehicle's movement. This demonstrates how perception systems used in autonomous driving can interpret traffic instructions and convert them into control actions.

The system is designed as a **low-cost prototype using a laptop webcam**, making it suitable for academic research and demonstration purposes.

This project was developed as part of a **B.Tech Capstone Project in Computer Science and Engineering at GITAM University, Bengaluru**. 

---

# 🎯 Objectives

The main objectives of this project are:

• Develop a deep learning model to recognize traffic signals and traffic police hand gestures
• Implement real-time signal detection using computer vision
• Integrate the detection system with an autonomous vehicle simulation
• Demonstrate how AI perception systems interpret traffic instructions
• Build a prototype for intelligent transportation research

---

# 🧠 Technologies Used

### Programming Language

* Python

### Deep Learning Framework

* TensorFlow
* Keras

### Computer Vision

* OpenCV

### Simulation

* Pygame

### Data Processing

* NumPy

### Development Tools

* Visual Studio Code
* Git & GitHub

---

# 🏗 System Architecture

The system follows a **three-layer architecture similar to real autonomous vehicle systems**:

### 1️⃣ Perception Layer

* Captures video frames using a webcam
* Detects traffic signals and hand gestures using a CNN model

### 2️⃣ Decision Layer

* Interprets predicted signals
* Determines vehicle actions such as stopping or lane change

### 3️⃣ Control Layer

* Updates vehicle speed and lane position in simulation

This layered architecture demonstrates how **perception, decision, and control modules interact in intelligent transportation systems**. 

---

# 🧠 Deep Learning Model

The model is built using **MobileNetV2 with Transfer Learning**.

### Key Features

* Lightweight architecture
* Pretrained on ImageNet
* Efficient for real-time inference
* Suitable for low-power systems

### Model Pipeline

1️⃣ Dataset preparation
2️⃣ Image preprocessing and augmentation
3️⃣ Transfer learning with MobileNetV2
4️⃣ Training with categorical classification
5️⃣ Model deployment for real-time prediction

The trained model is saved as:

```
traffic_hand_signal_cnn.h5
```

---

# 📂 Dataset Classes

The dataset contains **13 signal and gesture categories**:

```
0 - Green Light
1 - Red Light
2 - Yellow Light
3 - Lane Left
4 - Lane Right
5 - Left
6 - Left Over
7 - Left Turn
8 - Move Straight
9 - Right
10 - Right Over
11 - Right Turn
12 - Stop Signal
```

Images are organized into **class-specific folders for training**.

---

# 📁 Project Structure

```
Capstone-Project
│
├── DataSet/
│   ├── 0-Green Light
│   ├── 1-Red Light
│   ├── 2-Yellow Light
│   ├── ...
│   └── 12-stop signal
│
├── train_model.py
├── live_predict.py
├── car_simulation.py
├── traffic_hand_signal_cnn.h5
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Chandra1086/Capstone-Project.git
cd Capstone-Project
```

---

## 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```
pip install tensorflow opencv-python numpy pygame
```

---

# ▶️ Running the Project

## Train the Model

```
python train_model.py
```

---

## Real-Time Signal Detection

```
python live_predict.py
```

The system will:

• Capture webcam video
• Detect signals
• Display predicted class and confidence score

---

## Run Autonomous Vehicle Simulation

```
python car_simulation.py
```

The simulation includes:

• Circular multi-lane road
• Dynamic vehicle movement
• Real-time response to detected signals

Examples:

| Signal           | Vehicle Action              |
| ---------------- | --------------------------- |
| Red Light / Stop | Vehicle stops               |
| Green Light      | Vehicle moves               |
| Left Gesture     | Vehicle moves to left lane  |
| Right Gesture    | Vehicle moves to right lane |

---

# 📊 Features

✔ Real-time traffic signal detection
✔ Traffic police hand gesture recognition
✔ Deep learning using MobileNetV2
✔ Transfer learning for efficient training
✔ Autonomous vehicle simulation
✔ Real-time webcam processing
✔ Lane-changing vehicle behavior

---

# 🚀 Applications

• Intelligent Traffic Monitoring Systems
• Autonomous Vehicle Perception Systems
• Smart City Traffic Management
• Driver Assistance Systems
• AI-Based Transportation Research

---

# 🔮 Future Improvements

Future enhancements for this system include:

• Using **YOLO-based object detection models**
• Expanding dataset for improved accuracy
• Deploying on **embedded devices like Raspberry Pi**
• Integrating **real traffic camera feeds**
• Developing a complete autonomous driving environment

These improvements can transform the prototype into a **fully automated traffic recognition system**. 

---

# 👨‍💻 Authors

**A.V. Chandrakanth Reddy**
B.Tech Computer Science and Engineering
GITAM University, Bengaluru

Team Members:

* Anuguru Veera Chandrakanth Reddy
* Thammineni Karuna Sree
* Manasa GM
* Singareddy Sathish Reddy

Project Guide:

**Dr. Sreenarayanan N M**
Assistant Professor
Department of Computer Science and Engineering
GITAM School of CSE, Bengaluru 

---

# 📜 License

This project is developed for **academic and research purposes**.

1️⃣ `requirements.txt` for your project
2️⃣ **GitHub badges + project banner (AI / ML style)**.
