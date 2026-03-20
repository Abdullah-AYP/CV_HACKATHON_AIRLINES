<div align="center">
  
  <img src="demo.gif" alt="Safety Detection Live Demo" width="800">
  
  # 🚀 AI-Powered Safety Equipment Detection
  **High-Precision Instance Segmentation & Object Detection in Cluttered Environments**

  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black" alt="YOLOv8">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  </p>
</div>

---

### 👁️ Visual Results & Detections

| Environment | Model Detection Confidence |
| :---: | :---: |
| <img src="00000003_vcluttered_room.png" width="400" alt="Dark Cluttered"> | <img src="00000002_vlight_cluttered.png" width="400" alt="Detected"> |
| *Heavy Occlusion (First Aid Box)* | *High Confidence Detection (0.8+)* |
| <img src="008000001_vcluttered_hallway.png" width="400" alt="Hallway"> | <img src="000000001_vlight_undcluttered.png" width="400" alt="Detected"> |
| *Complex Angles (Oxygen Tanks)* | *Precision Bonding Boxes* |

---

### 🧠 Inference Pipeline (How it Works)
The system doesn't just pass images to a model. It dynamically adjusts resolution and utilizes Test Time Augmentation (TTA) to handle extreme domain shifts and lighting changes.

```mermaid
graph LR
    A[Raw Image/Stream] --> B[Upscale to 1024x1024]
    B --> C{YOLOv8 Large Engine}
    C --> D[Test Time Augmentation]
    C --> E[Agnostic NMS]
    D --> F[Confidence Filter]
    E --> F
    F --> G((Final Output Map))
    
    style C fill:#00FFFF,stroke:#000,stroke-width:2px,color:#000
```

---

### 📊 Performance Metrics & Engineering

Standard out-of-the-box training failed on small objects like distant fire alarms. To hit our **`0.7871` Test mAP@0.5**, we engineered the following solutions:

* 🎯 **Resolution Scaling:** Pushed inputs to `1024px` to capture micro-features, boosting small-object mAP by ~15%.
* 🌓 **Test Time Augmentation (TTA):** Processed images at multiple scales and flips during inference to counteract dim cabin lighting and weird camera angles.
* 📦 **Agnostic NMS:** Overrode standard suppression to ensure overlapping boxes of *different* classes (e.g., a Med Kit on a Chair) weren't accidentally deleted.
* 📉 **Cosine Annealing:** Dropped the learning rate aggressively in the final 5 epochs (`cos_lr=True`) to settle into a sharper local minimum.

---

### 🚀 Live Inference Quick Start

**1. Clone & Install:**
```bash
git clone [https://github.com/Abdullah-AYP/CV_HACKATHON_AIRLINES.git](https://github.com/Abdullah-AYP/CV_HACKATHON_AIRLINES.git)
cd CV_HACKATHON_AIRLINES
pip install ultralytics torch torchvision
```

**2. Run the Visualizer (Live Demo):**
```bash
yolo task=detect mode=predict model=weights/best.pt source=your_test_video.mp4 conf=0.20 agnostic_nms=True augment=True
```
*(Note: `conf=0.20` is tuned for a clean visual UI, while `conf=0.001` was used internally to maximize the PR-Curve).*

---
<div align="center">
  <i>Developed for a competitive Computer Vision Hackathon. Engineered for precision under pressure.</i>
</div>
