# Person_reID_baseline_pytorch-master

This document provides a comprehensive guide to the Person Re-Identification (Re-ID) and tracking pipeline. The system is designed to identify and track individuals across video frames using deep learning appearance embeddings.

---

## 1. Project Overview

The pipeline consists of four main stages:
1.  **Data Preparation**: Organizing raw person images into a structured format.
2.  **Training**: Learning a robust appearance model from large-scale pedestrian datasets.
3.  **Feature Extraction**: Converting person images or video frames into numerical vectors (embeddings).
4.  **End-to-End Tracking**: Combining object detection with Re-ID embeddings to maintain consistent identities over time.

---

## 2. Method

The system utilizes a deep learning architecture, specifically a ResNet-50 backbone fine-tuned on person re-identification datasets, to extract high-dimensional appearance embeddings from detected pedestrians. For tracking, object detection (YOLO) identifies persons in video frames, and their identity is maintained by calculating the Cosine Similarity between current embeddings and a gallery of previous trajectories. A Hungarian algorithm is employed for optimal identity matching, ensuring robust re-identification even after temporary occlusions or scene exits.

---

## 2. Setup & Installation

### Environment Configuration
1. Create a virtual environment to isolate dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the necessary libraries:
   ```bash
   pip install torch torchvision torchaudio
   pip install numpy scipy pyyaml tqdm opencv-python matplotlib
   pip install ultralytics timm
   ```

---

## 3. Data Preparation

Standard Re-ID datasets (like Market-1501) must be organized into a specific directory structure for processing.

1. Place your raw dataset (e.g., `Market-1501-v15.09.15`) in the project root.
2. Run the preparation script:
   ```bash
   python prepare.py
   ```
   This script creates a `pytorch` directory with subfolders for `train`, `val`, `query`, and `gallery`, sorting images by their Person ID.

---

## 4. Training the Model

To train a custom Re-ID model (e.g., ResNet50) on your structured data:

1. Start the training process:
   ```bash
   python train.py --name ft_ResNet50 --data_dir ./Market/pytorch --batchsize 32 --total_epoch 60
   ```
2. The script will save the best model weights to `./model/ft_ResNet50/net_last.pth`.
3. You can monitor the training progress via `train.jpg` generated in the model folder.

---

## 5. Standalone Feature Extraction

If you have a set of images and want to extract their identity embeddings without tracking:

1. Use the feature extraction script:
   ```bash
   python extract_features.py --data_dir ./Market/pytorch --model_name ft_ResNet50
   ```
2. This will output:
   - `feature_vectors.npy`: A matrix of 512-dimensional embeddings.
   - `image_filenames.npy`: The corresponding filenames for each vector.

---

## 6. End-to-End Pedestrian Tracking

The most advanced part of the pipeline combines real-time object detection with our Re-ID model to track people across a video.

### Tracking Logic
- **Detection**: Uses a high-performance detector (YOLO) to find people in every alternate frame.
- **Embedding**: Crops the detected people and passes them through the Re-ID model.
- **Matching**: Uses a Hungarian algorithm to match current detections with previous trajectories based on "Cosine Similarity."
- **Persistence**: Remembers identities for a long duration, allowing for tracking recovery even after significant occlusion or subjects leaving/re-entering the frame.

### Usage
Run the tracking script on any video file:
```bash
python track_and_reid.py --video_path "path/to/your/video.mp4" --output_video "path/to/result.mp4"
```

---

## 7. Results & Visualization

The output video will display bounding boxes and unique ID tags that stay locked to the same individual throughout the scene.

![Tracking Visualization Placeholder](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjB1ZXoydnVlYXd1dnZhcDM0OG43Y2JvMnk5Z2xnajY0MWJtODVqMyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/XwvwQpp1Xue9PV7PMQ/giphy.gif)

> [!TIP]
> **Performance Optimization**: The system is configured to process alternate frames, doubling execution speed while maintaining tracking consistency.
