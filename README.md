# Vehicle Monitoring System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/Object%20Detection-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-orange.svg)](https://github.com/JaidedAI/EasyOCR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, end-to-end traffic surveillance solution engineered to detect, track, and recognize vehicle license plates with high precision. This system is specifically optimized for **Gujarat-region vehicle formats**, featuring custom OCR post-processing and robust tracking logic.

---

## 🌟 Key Features

*   **🎯 High-Precision Detection:** Powered by a custom-trained YOLOv8 model for reliable plate localization even in varied lighting.
*   **🔄 Persistent Tracking:** Implements ByteTrack-based IDs to ensure each vehicle is logged exactly once as it passes through the detection zone.
*   **📸 Advanced Image Preprocessing:** Automatic plate extraction, cubic 4x scaling, grayscale conversion, and Laplacian kernel sharpening for superior OCR accuracy.
*   **🔡 Smart OCR Correction:** Custom logic to handle common mischaracterizations (e.g., correcting "6/G/C" prefixes to "GJ" for Gujarat plates).
*   **📊 Automated Logging:** Generates a real-time `traffic_logs.csv` with timestamps, plate numbers, and entry/exit statuses.
*   **🛡️ Hardware Optimized:** Full support for CUDA-accelerated inference for both YOLOv8 and EasyOCR.

---

## 🛠️ Technical Workflow

The system follows a modular pipeline to ensure maximum reliability:

1.  **Frame Capture:** Reads high-resolution streams from `test_videos/`.
2.  **Object Tracking:** YOLOv8 detects plates and assigns a unique `track_id`.
3.  **Zone Validation:** Only processes plates within the defined `ZONE_TOP` (300) and `ZONE_BOTTOM` (950) markers.
4.  **Feature Extraction:** Crops the bottom 50% of the detected plate area to minimize background noise.
5.  **OCR Pipeline:** Preprocessed crops are sent to the EasyOCR engine for text extraction.
6.  **Data Persistence:** Validated plates are logged to CSV and tracked via a local cache to prevent duplicate entries.

---

## 📂 Project Architecture

```text
.
├── main.py                 #  The core execution engine
├── train.py                #  Model training pipeline
├── test.py                 #  Hardware & Environment validator
├── data.yaml               #  Dataset & Class configuration
├── traffic_logs.csv        #  Live traffic data records
├── runs/                   #  Training artifacts & weights
│   └── detect/train/weights/best.pt
├── images/                 #  Training/Validation dataset
└── test_videos/            #  Source footage for testing
```

---

## 🚀 Quick Start

### 1. Environment Setup
Clone the repository and install the required deep learning dependencies:
```bash
git clone https://github.com/yourusername/vehicle-monitoring-system.git
cd vehicle-monitoring-system
pip install -r requirements.txt
# Alternatively:
pip install ultralytics easyocr opencv-python torch numpy
```

### 2. Verify Hardware
Ensure your GPU is detected and the OCR engine is ready:
```bash
python test.py
```

### 3. Run Inference
Launch the monitoring system on the default video source:
```bash
python main.py
```

---

## 🏋️ Training Guide

To retrain the model on the **Gujarat Vehicle Dataset**:
1. Update `data.yaml` with your absolute paths.
2. Execute the training script:
   ```bash
   python train.py
   ```
*The script is pre-configured for 50 epochs with a batch size of 4, optimized for consumer-grade GPUs.*

---

## 📈 Performance Tuning

You can fine-tune the system's sensitivity in `main.py`:
- **`conf=0.3`**: Increase this value to reduce false positives.
- **`ZONE_TOP/BOTTOM`**: Adjust these to match your camera's angle and lane position.
- **`preprocess_plate()`**: Modify the sharpening kernel if dealing with motion blur.

---

## 🗺️ Roadmap & Future Enhancements

- [ ] **Speed Estimation:** Calculate vehicle velocity based on frame-to-frame pixel displacement.
- [ ] **Vehicle Classification:** Distinguish between 2-wheelers, 4-wheelers, and heavy vehicles.
- [ ] **Cloud Sync:** Automated upload of `traffic_logs.csv` to a centralized dashboard.
- [ ] **Web UI:** A React/FastAPI-based dashboard for real-time visualization.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---
**Developed by Prem Shah for college use.**
