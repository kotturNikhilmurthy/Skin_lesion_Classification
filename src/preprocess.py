import cv2
import numpy as np


def remove_black_borders(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove black borders from skin lesion images.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find non-black regions
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add small padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)

        # Crop
        img = img[y:y + h, x:x + w]

    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Enhanced medical image preprocessing for skin lesions:
    - Remove black borders
    - CLAHE on L channel (adaptive histogram equalization)
    - Gentle color enhancement
    - Edge-preserving filtering

    Args:
        img (np.ndarray): RGB image, uint8, shape (H, W, 3)

    Returns:
        np.ndarray: Preprocessed RGB image, uint8
    """

    # Ensure correct type
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Safety check
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected RGB image with 3 channels")

    # Remove black borders (common in dermatoscopic images)
    img = remove_black_borders(img)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to L channel (adaptive contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Gentle brightness adjustment (reduced from 1.2 to 1.05)
    img = np.clip(img.astype(np.float32) * 1.05, 0, 255).astype(np.uint8)

    # Edge-preserving filter (bilateral filter)
    # This reduces noise while keeping edges sharp
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    return img