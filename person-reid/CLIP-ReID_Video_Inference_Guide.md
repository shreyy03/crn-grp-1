# CLIP-ReID Video Inference & Tracking

This document provide a comprehensive guide to the **CLIP-based** Person Re-Identification (Re-ID) and tracking pipeline. Unlike traditional Re-ID models, this system leverages **Vision Transformers (ViT)** and **CLIP** embeddings to achieve superior identity consistency across video frames.

---

## 1. Project Overview

The pipeline implements an advanced tracking-by-detection framework:
1.  **Person Detection**: Using Faster R-CNN with a ResNet-50 backbone.
2.  **Appearance Embedding**: Using CLIP-ViT-B-16 with **Side Information Embedding (SIE)** for camera-aware normalization.
3.  **Global Identity Matching**: A greedy global-best-match algorithm with **L2-normalized cosine similarity**.
4.  **Conflict Resolution**: Preventing identity duplication within a single frame through a non-duplicate assignment logic.

---

## 2. Method

The core of this system is the **CLIP-ReID** model, which translates pedestrian images into 768-dimensional feature vectors.

### Key Technical Enhancements:
- **Vision Transformer (ViT)**: Captures global context, making it more resilient to occlusions compared to standard CNNs.
- **SIE (Side Information Embedding)**: Explicitly embeds camera ID information, allowing the model to normalize for camera-specific color shifts (e.g., different lighting in different views).
- **L2 Normalization**: Ensures that feature vectors have a unit length, allowing for precise 0.0 to 1.0 similarity scoring through dot products.
- **Conflict Management**: If two people in a frame look like "Person 1," the system assigns the ID to the most confident match and assigns a new ID to the other, ensuring 100% unique identity allocation per frame.

---

## 3. Setup & Installation

### Environment Requirements
1. Ensure Python 3.8+ is installed.
2. Install the core dependencies:
   ```bash
   pip install torch torchvision timm yacs opencv-python pillow
   ```
3. **Model Weights**: Place the CLIP-ReID pretrained weights (e.g., `Market1501_clipreid_12x12sie_ViT-B-16_60.pth`) in the `checkpoints/` directory.

---

## 4. Data Preparation

CLIP-ReID requires datasets to be organized in a standard format (e.g., Market-1501, MSMT17).

1.  Download and unzip your dataset to a local directory.
2.  Update the `configs/person/vit_clipreid.yml` (or your chosen config) to point to the dataset:
    ```yaml
    DATASETS:
      NAMES: ('market1501')
      ROOT_DIR: ('path/to/your/datasets')
    OUTPUT_DIR: 'runs/market1501_vit'
    ```

---

## 5. Training the Model

The training process for CLIP-ReID typically involves a two-stage optimization (Stage 1 for text-related prompts and Stage 2 for the full model).

### Basic Training
To start training with default parameters:
```bash
python train_clipreid.py --config_file configs/person/vit_clipreid.yml
```

### Advanced Training (with SIE)
To train with **Side Information Embedding (SIE)** and optimized stride:
```bash
python train_clipreid.py \
    --config_file configs/person/vit_clipreid.yml \
    MODEL.SIE_CAMERA True \
    MODEL.SIE_COE 1.0 \
    MODEL.STRIDE_SIZE '[12, 12]'
```
- **Weights**: Trained models will be saved in the `OUTPUT_DIR` specified in your config file (e.g., `runs/market1501_vit/ViT-B-16_60.pth`).

---

## 6. Video Preprocessing

For high-resolution videos, use the sampling script to increase processing speed and reduce GPU/CPU load:

```bash
python preprocess_video.py --input test-video/input.mp4 --output test-video/sampled.mp4 --interval 2
```
*This reduces the frame count by 50% while maintaining enough temporal data for consistent ReID tracking.*

---

## 7. End-to-End Inference

The `inference_video.py` script performs the full detection and Re-ID matching process.

### Tracking Logic
- **Robust Similarity (0.85 Threshold)**: A strict threshold prevents "ID drifting" where person B is mistaken for person A.
- **Dynamic Gallery**: The system builds a "memory" of every unique person seen, allowing it to recognize someone if they leave and re-enter the frame.
- **Frame Skipping**: Optimized to process specific intervals if configured.

### Usage
```bash
python inference_video.py \
    --config_file configs/person/vit_clipreid.yml \
    --input "test-video/your_video.mp4" \
    --output "test-video/result.mp4" \
    --max_frames 100 \
    --opts TEST.WEIGHT checkpoints/Market1501_clipreid.pth
```

---

## 8. Results & Visualization

The output video will display:
- **Green Bounding Boxes**: For every detected person with confidence > 0.8.
- **ID Tags**: A persistent ID (e.g., "ID 1", "ID 2") that stays locked to the individual.
- **Score Logging**: Console output showing the matching confidence (e.g., `Match: ID 1 with score 0.945`).

![alt text](image.png)

![Tracking Visualization Placeholder](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExb2x3OWQwZ3EzMWoyeGd5MHVzbW4xeDVpamxkM3V2dW5hemV4bXVqaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Hg27IbDq6lpNKAwTeW/giphy.gif)

> [!TIP]
> **Performance Optimization**: Use the `--frame_interval` argument (e.g., `--frame_interval 2`) to process every N-th frame. This significantly speeds up inference on CPU without sacrificing matching quality.
