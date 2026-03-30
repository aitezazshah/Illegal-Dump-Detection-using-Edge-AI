# Illegal-Dump-Detection-using-Edge-AI
An AI-powered illegal dump detection system using YOLOv8 trained on 3,000+ black waste bag images. Deployed on Raspberry Pi 5 with live camera feed, it automatically detects Trashbags and persons in real-time, records evidence clips with timestamps, and saves proof of illegal dumping events for review.

# 🗑️ Illegal Dump Detection System
### Black Waste Bag Detection using YOLOv8 + Raspberry Pi 5

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-red?style=flat-square&logo=raspberrypi)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Problem Statement

Illegal dumping of black waste bags in public spaces, alleyways, and restricted areas is a growing environmental and public health concern. Manual monitoring of such areas is expensive, labour-intensive, and impractical at scale. There is a need for an automated, real-time system that can detect illegal dumping events, capture evidence, and raise alerts without requiring constant human supervision.

---

## 💡 Solution

This project presents an end-to-end **computer vision pipeline** that:

- Trains a **YOLOv8s** model on a custom black waste bag dataset
- Detects **Trashbags** and **Persons** simultaneously in real-time
- Runs on a **Raspberry Pi 5** using a live camera feed
- Automatically **records an evidence clip** when an illegal dump event is detected
- Saves timestamped evidence files locally for review

```
┌─────────────────────────────────────────────────────────┐
│                   SYSTEM PIPELINE                       │
│                                                         │
│  Camera Feed → YOLOv8s (Trashbag) ─┐                   │
│                                     ├→ Event Detected   │
│              → YOLOv8n (Person)   ─┘        ↓          │
│                                         Extract Clip   │
│                                              ↓          │
│                                     Save Evidence File  │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Dataset Details

| Property | Details |
|---|---|
| Dataset Name | Black_Waste_Bag |
| Source | Roboflow |
| Total Images | ~3,073 |
| Classes | 1 — `Trashbag` |
| Format | YOLOv8  |
| Train Split | ~2,152 images (70%) |
| Validation Split | ~615 images (20%) |
| Test Split | ~306 images (10%) |
| Augmentations | Mosaic, Flip LR, HSV shifts |

### Dataset Directory Layout after extraction:
```
/content/dataset/
    train/images/     ← 2,152 .jpg files
    train/labels/     ← matching .txt annotation files
    valid/images/
    valid/labels/
    test/images/
    test/labels/
```

---

## 🧠 Model Architecture

### Two-Model Pipeline

| Model | Purpose | Trained On | Classes | File |
|---|---|---|---|---|
| YOLOv8s | Trashbag Detection | Custom 3k image dataset | 1 (Trashbag) | `best.onnx` |
| YOLOv8n | Person Detection | COCO (330k images) | 80 (we use class 0-Person) | `yolov8n.pt` |

**Why two models?**

- `YOLOv8s` was fine-tuned on your dataset and **forgot** all COCO classes — it only knows `Trashbag`
- `YOLOv8n` is the original pretrained COCO model that already knows `person` very reliably
- Both run on every frame and results are drawn together

### Training Hyperparameters

```python
MODEL_SIZE    = 'yolov8s'
EPOCHS        = 50
IMGSZ         = 640
BATCH         = 16
PATIENCE      = 20          # early stopping
LR0           = 0.01
WARMUP_EPOCHS = 3
MOSAIC        = 1.0
FLIPLR        = 0.5
HSV_S         = 0.7
HSV_V         = 0.4
```

---

## 📊 Results

### Confusion Matrix

```
                    TRUE
                Trashbag    Background
           ┌────────────┬────────────┐
PREDICTED  │    1750    │    512     │  ← Trashbag
           ├────────────┼────────────┤
           │    307     │     0      │  ← Background
           └────────────┴────────────┘

  1750 → True Positive  ✅ Correct detection
   512 → False Positive ❌ False alarm (background mistaken for bag)
   307 → False Negative ❌ Missed bag (real bag not detected)
```

### Evaluation Metrics

| Metric | Training | Validation | Test |
|---|---|---|---|
| Precision | 91.1% | 82.3% | **82.9%** |
| Recall | 91.5% | 79.4% | **82.8%** |
| F1 Score | 91.3% | 80.9% | **82.9%** |
| mAP@0.5 | 96.0% | 86.2% | **88.2%** |
| mAP@0.5:0.95 | 65.9% | 51.3% | **52.4%** |
| Accuracy | 84.1% | 68.4% | **70.8%** |

### Metric Definitions

- **Precision:** Of all bags the model detected, how many were actually real bags?
- **Recall:** Of all real bags that existed, how many did the model find?
- **F1-Score:** Balance between Precision and Recall combined into one single score.
- **mAP@0.5:** Was the bag detected AND was the box placed in roughly the right spot?
- **mAP@0.5:0.95:** Same as mAP@0.5 but judges box tightness much more strictly.
- **Accuracy:** Out of all predictions made, how many were correct overall?

### Training Curves Summary

All three losses (`box_loss`, `cls_loss`, `dfl_loss`) decreased consistently across 50 epochs on both training and validation sets — confirming healthy learning with no overfitting. Precision and Recall both climbed from ~0.48 at epoch 1 to ~0.83 by epoch 50. mAP@0.5 reached **0.88** on the test set.

---

## 🖥️ Hardware Setup

### Raspberry Pi 5 Configuration

```
Hardware Required:
┌─────────────────────────────────────────┐
│  Raspberry Pi 5 (4GB or 8GB RAM)        │
│  MicroSD Card (32GB+ recommended)       │
│  Pi Camera Module v2 / USB Webcam       │
│  Power Supply (5V 5A USB-C)             │
│  Monitor, keyboard, mouse               │
└─────────────────────────────────────────┘
```


---

## 🚀 Running the Project

### Part 1 — Training on Google Colab

#### Step 1: Open the notebook
```
Upload .ipynb file to Google Colab
Runtime → Change runtime type → T4 GPU
```

#### Step 2: Upload dataset
```python
# Upload Black_Waste_Bag.zip when prompted
# Or connect to Roboflow directly
```

#### Step 3: Fix dataset paths
```python
import yaml
with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)

cfg['train'] = '/content/dataset/train/images'
cfg['val']   = '/content/dataset/valid/images'
cfg['test']  = '/content/dataset/test/images'

with open(YAML_PATH, 'w') as f:
    yaml.dump(cfg, f)
```

#### Step 4: Run all cells in order
```
0. GPU Check
1. Install dependencies
2. Upload & extract dataset
3. Validate & visualise dataset
4. Configure & train YOLOv8
5. Evaluate results
6. Export best weights
7. Video inference
```

#### Step 5: Download trained models immediately after training
```python
from google.colab import files
files.download('/content/runs/waste_bag/train_v1/weights/best.pt')

# Export and download ONNX for Pi
from ultralytics import YOLO
model = YOLO('/content/runs/waste_bag/train_v1/weights/best.pt')
model.export(format='onnx', imgsz=640, simplify=True, opset=12)
files.download('best.onnx')

# Download person model
person = YOLO('yolov8n.pt')
person.export(format='onnx', imgsz=640, simplify=True)
files.download('yolov8n.onnx')
```

### Part 2 — Deployment on Raspberry Pi 5

#### Step 1: Install dependencies
```bash
pip install ultralytics onnxruntime opencv-python
```

#### Step 2: Copy files to Pi
```bash
# From your laptop via SCP
scp best.onnx yolov8n.onnx waste_bag_detector.py pi@<PI_IP>:~/waste-bag-detector/
```

#### Step 3: Run the detector
```bash
cd ~/waste-bag-detector
python waste_bag_detector.py
```

#### Step 4: View saved evidence
```bash
ls evidence/
# illegal_dump_20250327_143022.mp4
# illegal_dump_20250328_091544.mp4
```

#### Step 5: Run headless (no monitor on Pi)
```python
# In waste_bag_detector.py — comment out this line:
# cv2.imshow('Waste Bag Detector', annotated_live)
# cv2.waitKey(1)
```

Then run as background service:
```bash
nohup python waste_bag_detector.py &
```

---

## ⚙️ Configuration Options

### Confidence Threshold Tuning

```python
BAG_CONF    = 0.3   # Lower → fewer missed bags, more false alarms
                    # Higher → fewer false alarms, more missed bags
PERSON_CONF = 0.4   # Person detection confidence
```


### Camera Source
```python
CAMERA_INDEX = 0     # 0 = default camera (Pi cam or USB)
                     # 1 = second camera if two are connected
```

---

## 🔍 Understanding the Output

### Live Feed Annotations

```
┌──────────────────────────────────────┐
│ ILLEGAL DUMP EVENT [REC]  14:32:05   │  ← Red banner when recording
├──────────────────────────────────────┤
│                                      │
│    ┌─────────┐  GREEN BOX = Person   │
│    │ Person  │                       │
│    │  0.87   │  RED BOX   = Trashbag │
│    └─────────┘                       │
│         ┌───────────────┐            │
│         │  Trashbag     │            │
│         │    0.74       │            │
│         └───────────────┘            │
└──────────────────────────────────────┘
```

-

## 🐛 Common Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: train/images` | Wrong path in data.yaml | Fix paths with the yaml patch cell |
| `0 proof frames saved` | Class name mismatch | Check `model.names` — use exact name |
| Empty sample images grid | Wrong image directory | Hardcode `/content/dataset/train/images` |
| JSON SyntaxError in notebook | Corrupted .ipynb file | Use the fixed version of the notebook |
| OOM on T4 GPU | Batch too large | Reduce `BATCH = 8` |
| Slow inference on Pi | Two models too heavy | Lower resolution or skip frames |

---

## 📈 Model Limitations & Future Work

**Current Limitations:**
- Model struggles with black bags on dark backgrounds (similar color/texture)
- Small bags in busy scenes can be missed
- mAP@0.5:0.95 is moderate — bounding boxes not always perfectly tight
- Two models running simultaneously is heavy for Pi CPU

**Future Improvements:**
- Add hard negative images (dark bins, black objects) to reduce false alarms
- Train for more epochs (70-100) for better convergence
- Use YOLOv8m for higher accuracy if Pi performance allows
- Add email/SMS alert when dump event detected
- Add GPS timestamp if Pi has GPS module
- Train person detection on your own dataset for better contextual awareness

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Model Architecture | YOLOv8s / YOLOv8n (Ultralytics) |
| Training Platform | Google Colab (T4 GPU) |
| Dataset Management | Roboflow |
| Deployment Hardware | Raspberry Pi 5 |
| Inference Runtime | ONNX Runtime |
| Video Processing | OpenCV |
| Language | Python 3.10+ |

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — model architecture and training framework
- [Roboflow](https://roboflow.com) — dataset hosting and augmentation
- [COCO Dataset](https://cocodataset.org) — pretrained weights for person detection

---

*Built for real-world illegal dump detection and evidence capture using edge AI on Raspberry Pi 5.*
