# suspicious_activity_baseline-master

This document provides a comprehensive guide to the Suspicious Activity / Violence Detection pipeline. The system is designed to classify scenes in images and video frames as violent or non-violent using a zero-shot deep learning approach powered by OpenAI's CLIP model.

---

## 1. Project Overview

The pipeline consists of four main stages:
1.  **Configuration**: Defining scene labels and model parameters in a YAML settings file.
2.  **Model Loading**: Initializing the CLIP (ViT-B/32) vision-language model and pre-computing text embeddings for all configured labels.
3.  **Prediction**: Encoding input images into visual embeddings and matching them against the text embeddings via Cosine Similarity to classify the scene.
4.  **Deployment**: Serving predictions through a CLI tool, a Jupyter Notebook (Google Colab), or direct Python API integration.

---

## 2. Method

The system utilizes OpenAI's CLIP (Contrastive Language-Image Pretraining) model with a Vision Transformer (ViT-B/32) backbone to perform zero-shot image classification. Instead of training on labeled violence datasets, descriptive text labels (e.g., "fight on a street", "fire in office") are encoded into text embeddings at initialization. At inference time, the input image is encoded into a visual embedding, and Cosine Similarity is computed between the image embedding and all pre-computed text embeddings. The label with the highest similarity score is selected, provided it exceeds a configurable confidence threshold; otherwise, a default "Unknown" label is returned. This approach allows the model to generalize to new scenarios simply by adding descriptive text labels to the configuration file—no retraining required.

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
   pip install -r requirements.txt
   ```
   Key dependencies include:
   ```bash
   pip install torch numpy matplotlib opencv-python Pillow
   pip install clip-by-openai pyaml
   ```

---

## 3. Configuration

Scene labels and model parameters are managed through the `settings.yaml` file. This is the primary way to customize the system for your use case.

1. Open `settings.yaml` to view or modify the configuration:
   ```yaml
   model-settings:
     prediction-threshold: 0.23   # Cosine similarity threshold (max 1.0)
     model-name: 'ViT-B/32'       # CLIP model variant
     device: 'cpu'                 # 'cpu' or 'cuda'
   label-settings:
     labels:
       - 'fight on a street'
       - 'fire on a street'
       - 'car crash'
       - 'violence in office'
       # ... add or remove labels as needed
     default-label: 'Unknown'
   ```
2. The model currently supports `16+1` scene labels (16 defined + 1 default "Unknown") covering both outdoor scenarios (fights, fires, car crashes) and indoor scenarios (office violence, office fires).
3. To detect a new scenario, simply add a descriptive text label to the `labels` list—no retraining is needed.

---

## 4. Running Predictions (CLI)

To classify a single image from the command line:

1. Run the prediction script:
   ```bash
   python run.py --image-path ./data/7.jpg
   ```
2. The script will:
   - Load the CLIP model and configured labels.
   - Read the specified image with OpenCV.
   - Output the predicted scene label to the console and display the image in a window.

---

## 5. Testing via Notebook (Google Colab)

The `suspicious_activity.ipynb` notebook provides a complete end-to-end testing pipeline designed for Google Colab with GPU acceleration (T4).

### Notebook Workflow
1. **Clone & Setup**: Clones the repository and installs dependencies (including CLIP from OpenAI's GitHub).
2. **Verify Configuration**: Inspects `settings.yaml` and confirms the CLIP ViT-B/32 model loads correctly (weights are downloaded automatically).
3. **Video Inference**: Processes a test video (`office_fight.mp4`) frame-by-frame using the `Model` class:
   ```python
   from model import Model
   import cv2

   model = Model()
   cap = cv2.VideoCapture('data/office_fight.mp4')
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break
       frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       prediction = model.predict(image=frame_rgb)
       label = prediction['label']
       confidence = prediction['confidence']
       text = f"{label} ({confidence:.2f})"
       cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
   ```
4. **Display Results**: The annotated output video is saved as `output_analysis.mp4` and rendered inline in the notebook using base64-encoded HTML video embedding.

### Usage
Open the notebook in Google Colab and run all cells:
```
suspicious_activity.ipynb
```

---

## 6. Python API Integration

To incorporate the violence detection model directly into your own project:

```python
from model import Model
import cv2

model = Model()
image = cv2.imread('./your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = model.predict(image=image)
print('Predicted label:', result['label'])
print('Confidence:', result['confidence'])
```

### Prediction Output
The `model.predict()` method returns a dictionary:
```python
{
    'label': 'fight on a street',   # Predicted scene label
    'confidence': 0.27              # Cosine similarity score
}
```

---

## 7. Results & Visualization

The system can process both images and videos frame-by-frame. Result videos display the model's predictions overlaid on each frame. Sample results are available in the `results/` directory.

![Result video](https://github.com/sagaryadavv/hr-report-dashboard/blob/main/output_fire.gif?raw=true)

![Result video](https://github.com/sagaryadavv/hr-report-dashboard/blob/main/output_fight.gif?raw=true)

> [!TIP]
> **Extending the Model**: To detect new suspicious scenarios (e.g., "shoplifting", "vandalism"), simply add a descriptive text label to `settings.yaml` under the `labels` key. The CLIP model's zero-shot capability allows it to generalize to new categories without any additional training.
