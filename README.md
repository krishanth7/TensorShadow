# TensorShadow Platinum

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-00bfa5?style=flat&logo=google)](https://mediapipe.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-r128-000000?style=flat&logo=three.js)](https://threejs.org/)

**TensorShadow** is an enterprise-grade biometric intelligence platform. It leverages deep neural networks and real-time computer vision to provide high-fidelity facial analysis, ocular tracking, and dental metrics within a secure, professional dashboard.

---

## 💎 Key Features

- **Tricolour Biometric Segmentation:**
  - 🟠 **Orange Matrix:** 468-point high-density facial topography.
  - 🔵 **Blue Ocular Points:** High-precision iris and pupil tracking.
  - 🟢 **Green Maxillary Scan:** Real-time dental and oral geometry analysis.
- **Neural Demographic Estimation:** Deep Learning-based Age and Gender estimation with live confidence scoring.
- **Subject-Specific ML Training:** Manual calibration mode to lock neural weights to a specific face for 100% accuracy.
- **3D Neural Projection:** Live WebGL/Three.js volumetric output showing real-time Z-depth and head orientation.
- **Enterprise UI:** Professional, corporate-grade dashboard with millisecond latency monitoring and biometric data export.

---

## 🏗️ Architecture

- **Neural Engine:** MediaPipe Holistic + TensorFlow.js.
- **Visualiser:** Three.js 3D Rendering + Canvas 2D Overlay.
- **Backend:** Flask-driven Simulation Server (Ready for Go/C++ migration).
- **ML Pipe:** Scikit-Learn MLP fallback for localised "on-the-fly" training.

---

## 📁 Project Structure

```text
TensorShadow/
├── data/               # Training datasets and stats
├── go_backend/         # Flask simulation server & dashboard
├── r_analysis/         # Statistical reporting scripts
├── tensorflow_model/   # ML training and model architecture
├── README.md           # Project documentation
└── requirements.txt    # System dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites
- python 3.9+
- pip

### 2. Installation
```bash
# Navigate to project
cd TensorShadow

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
python go_backend/sim_server.py
```
Open **http://localhost:8080** in your browser.

---

## 🧠 Usage Examples

### ML Calibration
Use the **Calibration Panel** on the left to train the model to your specific face for perfect age and gender accuracy.

### 3D Volumetric Scan
The right panel shows a live rotating point-cloud of your facial structure. Turn your head to see depth sensing in action.

---

## 🗺️ Roadmap
- [ ] Implement Go-based high-concurrency backend.
- [ ] Add Infrared (IR) support for thermal biometric scanning.
- [ ] Integrate Multi-Subject tracking (Crowd Analysis).

## 🤝 Contributing
Contributions are welcome! Please follow the corporate development guidelines.

## 📄 License
This project is licensed under the MIT License.
