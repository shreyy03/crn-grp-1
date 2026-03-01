# BPBReID - Body Part-Based Person Re-Identification

This document provides a comprehensive guide to the **BPBReID** (Body Part-Based Person Re-Identification) pipeline. The system is designed to identify and track individuals by leveraging part-based features, enhancing robustness against occlusions and clothing variations.

---

## 1. Project Overview

The pipeline consists of four main stages:
1.  **Data Preparation**: Formatting datasets like Market-1501, MSMT17, or custom data for the `torchreid` framework.
2.  **Training**: Training the HRNet-32 based BPBReID model using classification and part-based constraints.
3.  **Evaluation**: Testing model performance using Rank-1 and mAP metrics on query/gallery sets.
4.  **Video Inference & Tracking**: Running the trained model on video streams to perform real-time person tracking.

---

## 2. Method

BPBReID utilizes a **High-Resolution Network (HRNet-32)** backbone. Unlike standard global-feature models, BPBReID extracts features for specific anatomical body parts (head, torso, legs, etc.) and combines them with a global descriptor. This "part-aware" approach allows the model to focus on visible regions even when parts of the person are hidden. The tracking component uses **Faster R-CNN** for detection and **Cosine Similarity** with a feature gallery for identity maintenance.

---

## 3. Setup & Installation

### Environment Configuration
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install -r requirements.txt
   pip install opencv-python matplotlib
   ```

---

## 4. Data Preparation

BPBReID supports standard Re-ID datasets. You should download and extract them into a `data/` directory.

- **Market-1501**: `data/market1501/`
- **MSMT17**: `data/msmt17/`

The `torchreid` framework automatically handles the internal structure of these datasets once the root path is provided in the configuration.

---

## 5. Training the Model

Training is handled by the `torchreid/scripts/main.py` entry point.

1. **Configure Training**: Edit a YAML file in `configs/bpbreid/` (e.g., `bpbreid_market1501_train.yaml`).
2. **Start Training**:
   ```bash
   python torchreid/scripts/main.py --config-file configs/bpbreid/bpbreid_market1501_train.yaml
   ```
3. **Weights**: Trained models are typically saved in `log/bpbreid/model.pth.tar`.

---

## 6. Evaluation

To evaluate a trained model on a test set (e.g., Market-1501):

1. **Run Testing**:
   ```bash
   python torchreid/scripts/main.py --config-file configs/bpbreid/bpbreid_market1501_test.yaml --test-only
   ```
2. The output will provide **Rank-1 Accuracy** and **Mean Average Precision (mAP)**.

---

## 7. Video Inference & Tracking

You can run end-to-end tracking on a video using the custom inference script developed for this project.

### Tracking Logic
- **Detection**: Uses `Faster R-CNN` to detect all persons in each frame.
- **Part-Based Embedding**: BPBReID extracts high-resolution part features for each detection.
- **Greedy Matching**: Detections are matched to the identity gallery using **L2-Normalized Cosine Similarity**.
- **Gallery Update**: Identities are updated using a moving average of features to account for pose changes.

### Usage
Run the specialized video inference script:
```bash
python inference_video_bpbreid.py
```
*Note: Ensure the model path and video path are updated inside the script.*

---

## 8. Results & Visualization

The output video `test-video/result_video.mp4` will show bounding boxes and unique ID tags.

> [!IMPORTANT]
> **Part Features**: BPBReID is particularly effective in crowded scenes where people often overlap. It uses the `conct` (concatenated) feature map for matching, which includes detailed spatial information.

> [!TIP]
> **CPU vs GPU**: While the system runs on CPU, GPU acceleration via CUDA is significantly faster (30-50x) and is recommended for real-time applications.
