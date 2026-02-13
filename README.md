# DermaScan AI — Skin Lesion Classification for Early Cancer Detection

A deep learning-based clinical decision support system that classifies dermatoscopic skin lesion images into **Melanoma**, **Basal Cell Carcinoma**, and **Benign** categories. The project includes a full ML training pipeline, a Flask REST API backend, and a React (Vite + TypeScript) frontend web application.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation Results](#evaluation-results)
- [Preprocessing](#preprocessing)
- [Inference](#inference)
- [Web Application](#web-application)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

DermaScan AI uses an **EfficientNetV2-M** model (via [timm](https://github.com/huggingface/pytorch-image-models)) fine-tuned on the **BCN_20k** dermatoscopic image dataset. The system provides:

- 3-class skin lesion classification (Melanoma, Basal Cell Carcinoma, Benign)
- Medical image preprocessing (black border removal, CLAHE, bilateral filtering)
- Advanced training with Focal Loss, Mixup, CutMix, EMA, and warmup cosine scheduling
- Test-Time Augmentation (8 geometric transforms) for robust inference
- Confidence and entropy thresholding to flag uncertain predictions
- A production-ready Flask API serving a React frontend

---

## Project Structure

```
Skin lesions/
├── app.py                          # Flask API server (backend)
├── main.py                         # Default entry point (placeholder)
├── Dockerfile                      # Docker image for deployment (HF Spaces)
├── requirements.txt                # Python dependencies (training)
├── requirements_hf.txt             # Python dependencies (Hugging Face deployment)
│
├── src/                            # Core ML source code
│   ├── config.py                   # Hyperparameters & paths (CFG class)
│   ├── dataset.py                  # SkinDataset (PyTorch Dataset with Albumentations)
│   ├── model.py                    # Model factory (EfficientNetV2-M, ensemble support)
│   ├── preprocess.py               # Medical image preprocessing pipeline
│   ├── train.py                    # Training loop (Focal Loss, Mixup, CutMix, EMA)
│   ├── evaluate.py                 # Evaluation with Advanced TTA, ROC curves, confusion matrix
│   ├── inference.py                # SkinLesionClassifier class for single/batch prediction
│   ├── monitor_training.py         # Real-time training monitor (run in separate terminal)
│   ├── plot_training_curves.py     # Detailed training curve visualization
│   └── Quick start.py             # Full pipeline runner (train → evaluate → visualize)
│
├── scripts/
│   └── prepare_dataset.py          # BCN_20k CSV → balanced train/val/test split
│
├── models/                         # Saved model weights & training history
│   ├── best_model.pth              # Best validation accuracy checkpoint
│   ├── best_model_ema.pth          # Best EMA model checkpoint
│   ├── final_model.pth             # Final epoch checkpoint
│   ├── final_model_ema.pth         # Final EMA checkpoint
│   ├── convnext_tiny_best.pth      # Previous ConvNeXt-Tiny checkpoint
│   ├── convnext_best.pth           # Previous ConvNeXt checkpoint
│   └── training_history.json       # Epoch-wise loss, accuracy, LR history
│
├── data/
│   ├── bcn_20k_train.csv           # Training metadata CSV
│   ├── bcn_20k_test.csv            # Test metadata CSV
│   ├── BCN_20k_train/              # Raw training images
│   ├── BCN_20k_test/               # Raw test images
│   └── balanced_dataset/           # Prepared balanced dataset
│       ├── train/                  #   ├── melanoma/
│       │                           #   ├── basal_cell_carcinoma/
│       │                           #   └── benign/
│       ├── val/                    #   (same class subfolders)
│       └── test/                   #   (same class subfolders)
│
├── reports/                        # Evaluation outputs
│   ├── evaluation_results.json     # Accuracy, classification report, per-class metrics
│   └── test_results.txt            # Classification report (text)
│
├── static/                         # Built frontend assets served by Flask
│   ├── index.html
│   ├── robots.txt
│   └── assets/
│       ├── index-BMu4yhtp.css
│       ├── index-lsqhJ9oC.js
│       └── Scene3D-BGbHyZDe.js
│
└── dermascan-ai-insights-main/     # Frontend source (React + Vite + TypeScript)
    └── dermascan-ai-insights-main/
        ├── package.json
        ├── vite.config.ts
        ├── tailwind.config.ts
        ├── tsconfig.json
        └── src/
            ├── App.tsx
            ├── main.tsx
            ├── pages/
            │   ├── Index.tsx       # Main page layout
            │   └── NotFound.tsx
            └── components/
                ├── HeroSection.tsx
                ├── UploadPanel.tsx          # Image upload & prediction UI
                ├── MetricsDashboard.tsx     # Validation metrics charts
                ├── DatasetInfo.tsx
                ├── InterpretabilitySection.tsx
                ├── EthicsDisclaimer.tsx
                ├── Navbar.tsx
                ├── Footer.tsx
                ├── Scene3D.tsx             # Three.js 3D background
                └── ui/                     # shadcn/ui component library
```

---

## Dataset

**Source:** BCN_20k dermatoscopic image dataset.

The raw dataset is processed by [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py) which:
1. Maps diagnosis codes to 3 classes: `MEL → melanoma`, `BCC → basal_cell_carcinoma`, `NV/BKL → benign`
2. Balances classes to **1,500 samples per class** (undersampling the majority)
3. Splits into **train / val / test** (80% / 10% / 10%) with stratification

| Split | Samples per Class | Total |
|-------|-------------------|-------|
| Train | ~1,200            | ~3,600|
| Val   | ~150              | ~450  |
| Test  | ~150              | ~450  |

---

## Model Architecture

| Setting                | Value                        |
|------------------------|------------------------------|
| **Backbone**           | EfficientNetV2-M (`tf_efficientnetv2_m` via timm) |
| **Pretrained**         | ImageNet                     |
| **Input Size**         | 384 × 384                    |
| **Num Classes**        | 3                            |
| **Dropout**            | 0.3                          |
| **Drop Path Rate**     | 0.2 (stochastic depth)       |

An ensemble mode is also available in [`src/model.py`](src/model.py) combining EfficientNetV2-M, ConvNeXt-Base, and Swin Transformer.

---

## Training Pipeline

Configured in [`src/config.py`](src/config.py) and executed by [`src/train.py`](src/train.py).

| Hyperparameter         | Value                        |
|------------------------|------------------------------|
| **Optimizer**          | AdamW (betas=0.9, 0.999)     |
| **Learning Rate**      | 1e-4                         |
| **Min LR**             | 1e-6                         |
| **Weight Decay**       | 0.01                         |
| **Epochs**             | 50                           |
| **Batch Size**         | 8                            |
| **Loss Function**      | Focal Loss (gamma=2.0)       |
| **Class Weights**      | [1.5, 1.5, 1.0] (melanoma, BCC, benign) |
| **Label Smoothing**    | 0.1                          |
| **Scheduler**          | Warmup (5 epochs) + Cosine Annealing |
| **Mixup Alpha**        | 0.2                          |
| **CutMix Alpha**       | 0.2                          |
| **Gradient Clipping**  | 1.0                          |
| **EMA Decay**          | 0.9995                       |
| **Mixed Precision**    | FP16 (torch.cuda.amp)        |

**Data Augmentation** (via Albumentations):
- Horizontal/Vertical flip, RandomRotate90, Transpose
- ShiftScaleRotate (shift=0.1, scale=0.2, rotate=45°)
- HueSaturationValue, RandomBrightnessContrast, CLAHE
- GaussNoise, GaussianBlur, MotionBlur
- ElasticTransform (simulating skin deformation)
- CoarseDropout (simulating occlusions)
- ImageNet normalization

---

## Evaluation Results

Evaluated with **8× Test-Time Augmentation** (original + flips + rotations + transpose) using the EMA model weights.

### Overall Metrics (from [`reports/evaluation_results.json`](reports/evaluation_results.json))

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 88.00% |
| Precision | 87.96% |
| Recall    | 88.00% |
| F1-Score  | 87.98% |

### Per-Class Performance

| Class                 | Precision | Recall | F1-Score | Accuracy |
|-----------------------|-----------|--------|----------|----------|
| Melanoma              | 86.09%    | 86.67% | 86.38%   | 86.67%   |
| Basal Cell Carcinoma  | 92.76%    | 94.00% | 93.38%   | 94.00%   |
| Benign                | 85.03%    | 83.33% | 84.18%   | 83.33%   |

---

## Preprocessing

The medical image preprocessing pipeline ([`src/preprocess.py`](src/preprocess.py)) applies:

1. **Black border removal** — Detects and crops dark borders common in dermatoscopic images
2. **CLAHE** — Contrast Limited Adaptive Histogram Equalization on the L channel (LAB color space)
3. **Brightness adjustment** — Gentle 5% brightness boost
4. **Bilateral filtering** — Edge-preserving noise reduction (d=9, sigmaColor=75, sigmaSpace=75)

---

## Inference

The [`src/inference.py`](src/inference.py) module provides a `SkinLesionClassifier` class:

```python
from src.inference import SkinLesionClassifier

classifier = SkinLesionClassifier(use_tta=True)
class_name, confidence, all_probs = classifier.predict("path/to/image.jpg", return_all_probs=True)

print(f"Prediction: {class_name}, Confidence: {confidence:.2%}")
```

**Features:**
- Automatically loads the best EMA model weights
- Optional 6× TTA (original + flips + rotations)
- Single image and batch prediction support

**CLI usage:**
```bash
python -m src.inference path/to/skin_image.jpg
```

---

## Web Application

### Backend — Flask API ([`app.py`](app.py))

The Flask server loads the EfficientNetV2-M model and serves both the API and the static frontend:

- Listens on `0.0.0.0:5000` (or `PORT` env variable)
- Serves the built React frontend from the `static/` directory
- Provides REST endpoints for image classification

**Safety mechanisms:**
- **Confidence threshold** (60%) — Rejects low-confidence predictions
- **Entropy threshold** (0.90 ratio) — Flags uniformly distributed predictions as uncertain
- Returns medical recommendations with severity levels per class

### Frontend — React + Vite + TypeScript

Source code in [`dermascan-ai-insights-main/`](dermascan-ai-insights-main/dermascan-ai-insights-main/). Built with:

- **React 18** + **TypeScript** + **Vite**
- **Tailwind CSS** + **shadcn/ui** component library
- **Recharts** for performance metric charts
- **Three.js** (`@react-three/fiber` + `@react-three/drei`) for 3D hero background
- **React Router** for routing

**Key pages/components:**
- **UploadPanel** — Drag-and-drop image upload, sends to `/api/predict`, displays results with confidence bars and risk severity
- **MetricsDashboard** — Displays model evaluation metrics (accuracy, precision, recall, F1) with interactive bar charts
- **InterpretabilitySection** — Attention heatmap visualization placeholder
- **DatasetInfo** — Technical methodology and dataset information cards
- **EthicsDisclaimer** — Medical disclaimer

**Build the frontend:**
```bash
cd dermascan-ai-insights-main/dermascan-ai-insights-main
npm install
npm run build
```
Copy the build output to the `static/` folder to be served by Flask.

---

## API Endpoints

| Method | Endpoint        | Description                           |
|--------|-----------------|---------------------------------------|
| GET    | `/`             | Serves the frontend HTML              |
| POST   | `/api/predict`  | Classify an uploaded skin lesion image |
| GET    | `/api/health`   | Health check (model status, device)   |
| GET    | `/api/info`     | Model metadata and capabilities       |

### POST `/api/predict`

**Request:** `multipart/form-data`
- `image` (file) — Skin lesion image (JPEG/PNG)
- `use_tta` (string, optional) — `"true"` or `"false"` (default: `"true"`)

**Response (confident):**
```json
{
  "success": true,
  "uncertain": false,
  "prediction": "Melanoma",
  "confidence": 0.92,
  "probabilities": {
    "Melanoma": 0.92,
    "Basal Cell Carcinoma": 0.05,
    "Benign": 0.03
  },
  "recommendation": {
    "severity": "High Risk",
    "action": "Immediate dermatologist consultation required",
    "description": "..."
  },
  "tta_used": true,
  "model": "tf_efficientnetv2_m"
}
```

**Response (uncertain):**
```json
{
  "success": true,
  "uncertain": true,
  "prediction": "Benign",
  "confidence": 0.45,
  "entropy_ratio": 0.92,
  "message": "The model could not confidently classify this image...",
  "probabilities": { ... }
}
```

---

## Deployment

### Docker (Hugging Face Spaces)

The project includes a [Dockerfile](Dockerfile) targeting Hugging Face Spaces:

```bash
docker build -t dermascan-ai .
docker run -p 7860:7860 dermascan-ai
```

The Dockerfile uses `python:3.10-slim`, installs OpenCV system dependencies, copies the model weights (`best_model_ema.pth`) and static frontend, and exposes port **7860**.

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended; CPU supported)
- Node.js 18+ (for frontend development only)

### 1. Clone & Setup Python Environment

```bash
cd "Skin lesions"
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm albumentations opencv-python numpy pandas scikit-learn matplotlib pillow tqdm seaborn flask flask-cors
```

### 2. Prepare the Dataset

Place the BCN_20k dataset files in `data/`, then run:

```bash
python scripts/prepare_dataset.py
```

This creates the balanced `data/balanced_dataset/` with train/val/test splits.

### 3. Train the Model

```bash
python -m src.train
```

Or use the full pipeline runner:

```bash
python "src/Quick start.py"
```

Optionally monitor training in a separate terminal:

```bash
python -m src.monitor_training
```

### 4. Evaluate

```bash
python -m src.evaluate
```

Generates confusion matrix, ROC curves, and classification report in `reports/`.

### 5. Run the Web App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

### Quick Start (Full Pipeline)

```bash
python "src/Quick start.py"
```

Runs training → evaluation → visualization sequentially with interactive prompts.

### Single Image Inference

```bash
python -m src.inference path/to/skin_image.jpg
```

### Plot Training Curves

```bash
python -m src.plot_training_curves
```

Generates detailed training analysis plots in `reports/`.

---

## Dependencies

### Training & Inference

| Package        | Purpose                                     |
|----------------|---------------------------------------------|
| torch          | Deep learning framework                     |
| torchvision    | Image transforms & pretrained models        |
| timm           | EfficientNetV2-M, ConvNeXt, Swin models    |
| albumentations | Advanced image augmentation pipeline        |
| opencv-python  | Image preprocessing (CLAHE, bilateral filter) |
| numpy          | Numerical operations                        |
| pandas         | Dataset CSV processing                      |
| scikit-learn   | Metrics, train/test split, ROC-AUC          |
| matplotlib     | Training curve plots                        |
| seaborn        | Confusion matrix heatmaps                   |
| pillow         | Image I/O                                   |
| tqdm           | Progress bars                               |

### Web Application (Backend)

| Package     | Purpose                    |
|-------------|----------------------------|
| flask       | REST API server            |
| flask-cors  | Cross-Origin Resource Sharing |

### Frontend (Node.js)

| Package              | Purpose                         |
|----------------------|---------------------------------|
| react                | UI framework                    |
| vite                 | Build tool & dev server         |
| typescript           | Type safety                     |
| tailwindcss          | Utility-first CSS               |
| shadcn/ui (radix-ui) | Accessible UI components        |
| recharts             | Data visualization charts       |
| three.js             | 3D hero section background      |
| react-router-dom     | Client-side routing             |

---

## License

This project is for educational and research purposes. Not intended for clinical diagnosis. Always consult a qualified dermatologist for medical decisions.

