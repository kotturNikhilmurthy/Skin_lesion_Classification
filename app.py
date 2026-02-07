"""
Flask Backend API for Skin Lesion Classification
Handles image uploads, preprocessing, and model inference
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import base64
from pathlib import Path
import timm

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for frontend communication


# Configuration
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_SIZE = 384
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    MODEL_NAME = "tf_efficientnetv2_m"
    NUM_CLASSES = 3
    CLASSES = ["Melanoma", "Basal Cell Carcinoma", "Benign"]
    LABEL_MAP = {"melanoma": 0, "basal_cell_carcinoma": 1, "benign": 2}
    MODEL_PATH = Path("models/best_model_ema.pth")
    CONFIDENCE_THRESHOLD = 0.60  # Reject predictions below this confidence
    ENTROPY_THRESHOLD = 0.90    # Reject if entropy ratio exceeds this (too uncertain)


CFG = Config()


# Preprocessing functions
def remove_black_borders(img, threshold=10):
    """Remove black borders from dermatoscopic images"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        img = img[y:y + h, x:x + w]

    return img


def preprocess_image(img):
    """Apply medical image preprocessing"""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Remove borders
    img = remove_black_borders(img)

    # CLAHE for contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Gentle brightness adjustment
    img = np.clip(img.astype(np.float32) * 1.05, 0, 255).astype(np.uint8)

    # Bilateral filtering (edge-preserving)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    return img


# Model loading
print("🔄 Loading model...")
model = timm.create_model(
    CFG.MODEL_NAME,
    pretrained=False,
    num_classes=CFG.NUM_CLASSES
)

if CFG.MODEL_PATH.exists():
    model.load_state_dict(torch.load(CFG.MODEL_PATH, map_location=CFG.DEVICE))
    print(f"✅ Model loaded from {CFG.MODEL_PATH}")
else:
    print(f"⚠️ Model not found at {CFG.MODEL_PATH}")

model = model.to(CFG.DEVICE)
model.eval()

# Transform for inference
transform = A.Compose([
    A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
    A.Normalize(mean=CFG.MEAN, std=CFG.STD),
    ToTensorV2()
])


def advanced_tta(model, image_tensor):
    """Test-Time Augmentation with 8 variations"""
    predictions = []

    with torch.no_grad():
        # 1. Original
        pred = model(image_tensor).softmax(1)
        predictions.append(pred)

        # 2. Horizontal flip
        pred = model(torch.flip(image_tensor, dims=[3])).softmax(1)
        predictions.append(pred)

        # 3. Vertical flip
        pred = model(torch.flip(image_tensor, dims=[2])).softmax(1)
        predictions.append(pred)

        # 4. Both flips
        pred = model(torch.flip(image_tensor, dims=[2, 3])).softmax(1)
        predictions.append(pred)

        # 5. Rotate 90
        pred = model(torch.rot90(image_tensor, k=1, dims=[2, 3])).softmax(1)
        predictions.append(pred)

        # 6. Rotate 180
        pred = model(torch.rot90(image_tensor, k=2, dims=[2, 3])).softmax(1)
        predictions.append(pred)

        # 7. Rotate 270
        pred = model(torch.rot90(image_tensor, k=3, dims=[2, 3])).softmax(1)
        predictions.append(pred)

        # 8. Transpose
        pred = model(torch.transpose(image_tensor, 2, 3)).softmax(1)
        predictions.append(pred)

    # Average all predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and return predictions"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)

        # Preprocess
        preprocessed = preprocess_image(image_np)

        # Transform
        transformed = transform(image=preprocessed)['image']
        image_tensor = transformed.unsqueeze(0).to(CFG.DEVICE)

        # Get TTA parameter (default: True)
        use_tta = request.form.get('use_tta', 'true').lower() == 'true'

        # Predict
        if use_tta:
            probs = advanced_tta(model, image_tensor)
        else:
            with torch.no_grad():
                probs = model(image_tensor).softmax(1)

        # Get results
        confidence, pred_idx = torch.max(probs, 1)
        conf_value = float(confidence.item())
        predicted_class = CFG.CLASSES[pred_idx.item()]

        # All probabilities
        all_probs = {
            CFG.CLASSES[i]: float(probs[0, i].item())
            for i in range(CFG.NUM_CLASSES)
        }

        # --- Confidence & Entropy thresholding ---
        # Entropy: measures how spread out the prediction is
        # Max entropy for 3 classes = log(3) ≈ 1.099
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).item()
        max_entropy = np.log(CFG.NUM_CLASSES)
        entropy_ratio = entropy / max_entropy  # 0 = certain, 1 = uniform/random

        is_uncertain = conf_value < CFG.CONFIDENCE_THRESHOLD or entropy_ratio > CFG.ENTROPY_THRESHOLD

        if is_uncertain:
            response = {
                'success': True,
                'uncertain': True,
                'prediction': predicted_class,
                'confidence': conf_value,
                'entropy_ratio': round(entropy_ratio, 3),
                'probabilities': all_probs,
                'message': 'The model could not confidently classify this image. '
                           'This may not be a recognizable skin lesion, or the image quality may be insufficient.',
                'tta_used': use_tta,
                'model': CFG.MODEL_NAME
            }
            return jsonify(response)

        # Medical recommendations based on prediction
        recommendations = {
            "Melanoma": {
                "severity": "High Risk",
                "color": "#EF4444",
                "action": "Immediate dermatologist consultation required",
                "description": "Melanoma is the most dangerous form of skin cancer. Early detection and treatment are critical."
            },
            "Basal Cell Carcinoma": {
                "severity": "Moderate Risk",
                "color": "#F59E0B",
                "action": "Schedule dermatologist appointment soon",
                "description": "Most common skin cancer but rarely spreads. Treatment is highly effective."
            },
            "Benign": {
                "severity": "Low Risk",
                "color": "#10B981",
                "action": "Monitor for changes, routine check-ups recommended",
                "description": "Non-cancerous lesion. Continue regular skin examinations."
            }
        }

        response = {
            'success': True,
            'uncertain': False,
            'prediction': predicted_class,
            'confidence': float(confidence.item()),
            'probabilities': all_probs,
            'recommendation': recommendations[predicted_class],
            'tta_used': use_tta,
            'model': CFG.MODEL_NAME
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': CFG.MODEL_PATH.exists(),
        'device': CFG.DEVICE,
        'model': CFG.MODEL_NAME
    })


@app.route('/api/info', methods=['GET'])
def info():
    """Return model information"""
    return jsonify({
        'model_name': 'Skin Lesion Classifier',
        'architecture': CFG.MODEL_NAME,
        'classes': CFG.CLASSES,
        'accuracy': '88.00%',
        'image_size': CFG.IMAGE_SIZE,
        'device': CFG.DEVICE,
        'features': [
            'Test-Time Augmentation (8x)',
            'Medical image preprocessing',
            'CLAHE contrast enhancement',
            'Bilateral filtering'
        ]
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🏥 SKIN LESION CLASSIFICATION API")
    print("=" * 70)
    print(f"Device: {CFG.DEVICE}")
    print(f"Model: {CFG.MODEL_NAME}")
    print(f"Classes: {', '.join(CFG.CLASSES)}")
    print(f"Model loaded: {CFG.MODEL_PATH.exists()}")
    print("=" * 70)
    print("\n🚀 Starting server at http://localhost:5000")
    print("📱 Open http://localhost:5000 in your browser\n")

    app.run(debug=True, host='0.0.0.0', port=5000)