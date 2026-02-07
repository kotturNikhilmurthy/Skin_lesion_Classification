import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path

from src.model import get_model
from src.config import CFG
from src.preprocess import preprocess


class SkinLesionClassifier:
    """
    Advanced skin lesion classifier with TTA support
    """

    def __init__(self, model_path=None, use_tta=True):
        """
        Initialize the classifier

        Args:
            model_path: Path to model weights (default: best_model_ema.pth)
            use_tta: Whether to use test-time augmentation
        """
        self.device = CFG.DEVICE
        self.use_tta = use_tta

        # Load model
        self.model = get_model().to(self.device)

        # Load weights
        if model_path is None:
            # Try EMA model first
            ema_path = CFG.MODEL_DIR / "best_model_ema.pth"
            regular_path = CFG.MODEL_DIR / "best_model.pth"

            if ema_path.exists():
                model_path = ema_path
                print("✅ Using EMA model")
            elif regular_path.exists():
                model_path = regular_path
                print("✅ Using regular model")
            else:
                raise FileNotFoundError("No trained model found!")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"✅ Model loaded from {model_path}")

        # Define transform
        self.transform = A.Compose([
            A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
            A.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2()
        ])

        # Class names
        self.idx_to_class = {v: k for k, v in CFG.LABEL_MAP.items()}

    def _tta_predict(self, image_tensor):
        """Apply test-time augmentation"""
        predictions = []

        with torch.no_grad():
            # Original
            pred = self.model(image_tensor).softmax(1)
            predictions.append(pred)

            # Horizontal flip
            pred = self.model(torch.flip(image_tensor, dims=[3])).softmax(1)
            predictions.append(pred)

            # Vertical flip
            pred = self.model(torch.flip(image_tensor, dims=[2])).softmax(1)
            predictions.append(pred)

            # Both flips
            pred = self.model(torch.flip(image_tensor, dims=[2, 3])).softmax(1)
            predictions.append(pred)

            # Rotate 90
            pred = self.model(torch.rot90(image_tensor, k=1, dims=[2, 3])).softmax(1)
            predictions.append(pred)

            # Rotate 270
            pred = self.model(torch.rot90(image_tensor, k=3, dims=[2, 3])).softmax(1)
            predictions.append(pred)

        # Average predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred

    def predict(self, image_path, return_all_probs=False):
        """
        Predict class for a single image

        Args:
            image_path: Path to image file
            return_all_probs: If True, return probabilities for all classes

        Returns:
            If return_all_probs=False: (class_name, confidence)
            If return_all_probs=True: (class_name, confidence, all_probs_dict)
        """
        # Load and preprocess image
        img = np.array(Image.open(image_path).convert("RGB"))
        img = preprocess(img)

        # Transform
        transformed = self.transform(image=img)["image"]
        image_tensor = transformed.unsqueeze(0).to(self.device)

        # Predict
        if self.use_tta:
            probs = self._tta_predict(image_tensor)
        else:
            with torch.no_grad():
                probs = self.model(image_tensor).softmax(1)

        # Get prediction
        confidence, pred_idx = torch.max(probs, 1)
        class_name = self.idx_to_class[pred_idx.item()]

        if return_all_probs:
            all_probs = {
                self.idx_to_class[i]: probs[0, i].item()
                for i in range(len(CFG.CLASSES))
            }
            return class_name, confidence.item(), all_probs
        else:
            return class_name, confidence.item()

    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images

        Args:
            image_paths: List of image paths

        Returns:
            List of (class_name, confidence) tuples
        """
        results = []
        for img_path in image_paths:
            try:
                class_name, confidence = self.predict(img_path)
                results.append((class_name, confidence))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append((None, 0.0))

        return results


def predict_one_image(image_path, use_tta=True):
    """
    Convenience function for single image prediction

    Args:
        image_path: Path to image
        use_tta: Whether to use test-time augmentation

    Returns:
        class_name, confidence
    """
    classifier = SkinLesionClassifier(use_tta=use_tta)
    return classifier.predict(image_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    else:
        print("Usage: python -m src.inference <path_to_image>")
        print("\nExample:")
        print("  python -m src.inference data/test/melanoma/sample.jpg")
        sys.exit(1)

    if not Path(test_img).exists():
        print(f"❌ Image not found: {test_img}")
        sys.exit(1)

    print(f"\n🔍 Analyzing: {test_img}")
    print("-" * 60)

    # Predict with TTA
    classifier = SkinLesionClassifier(use_tta=True)
    class_name, confidence, all_probs = classifier.predict(
        test_img,
        return_all_probs=True
    )

    print(f"\n🎯 Prediction: {class_name}")
    print(f"📊 Confidence: {confidence * 100:.2f}%")
    print("\n📋 All probabilities:")
    for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cls:25s}: {prob * 100:5.2f}%")
    print("-" * 60)