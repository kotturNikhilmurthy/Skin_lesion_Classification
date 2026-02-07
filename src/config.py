import torch
from pathlib import Path

# ✅ Ensure this path is correct for your PC
DATA_ROOT = Path(r"C:\Users\kakhi\OneDrive\Desktop\Skin lesions\data\balanced_dataset")


class CFG:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Image Settings - INCREASED for better feature extraction
    IMAGE_SIZE = 384  # ⬆️ Increased from 224 to 384
    BATCH_SIZE = 8  # ⬇️ Reduced due to larger image size
    NUM_WORKERS = 0
    PIN_MEMORY = True

    # Training Settings - OPTIMIZED
    EPOCHS = 50  # ⬆️ Increased for convergence
    LR = 1e-4  # ⬇️ Reduced from 3e-4 for stability
    MIN_LR = 1e-6  # Minimum LR for scheduler
    WARMUP_EPOCHS = 5  # NEW: Warmup period
    WEIGHT_DECAY = 0.01  # ⬇️ Reduced for better generalization

    # Augmentation & Regularization
    MIXUP_ALPHA = 0.2  # NEW: Mixup augmentation
    CUTMIX_ALPHA = 0.2  # NEW: CutMix augmentation
    LABEL_SMOOTHING = 0.1  # NEW: Label smoothing

    # Class weights for focal loss (adjust based on your data distribution)
    CLASS_WEIGHTS = [1.5, 1.5, 1.0]  # [melanoma, basal_cell, benign]

    NUM_CLASSES = 3
    CLASSES = ["melanoma", "basal_cell_carcinoma", "benign"]

    LABEL_MAP = {
        "melanoma": 0,
        "basal_cell_carcinoma": 1,
        "benign": 2
    }

    TRAIN_DIR = DATA_ROOT / "train"
    VAL_DIR = DATA_ROOT / "val"
    TEST_DIR = DATA_ROOT / "test"

    MODEL_DIR = Path(__file__).parent.parent / "models"
    PLOTS_DIR = Path(__file__).parent.parent / "reports"

    # ImageNet normalization
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    # Model selection - using more powerful model
    MODEL_NAME = "tf_efficientnetv2_m"  # ⬆️ Upgraded from convnext_tiny
    # Alternative powerful models (uncomment to try):
    # MODEL_NAME = "convnext_base"
    # MODEL_NAME = "swin_base_patch4_window7_224"
    # MODEL_NAME = "vit_base_patch16_224"

    # Gradient clipping for stability
    GRAD_CLIP = 1.0

    # TTA settings
    TTA_TRANSFORMS = 8  # Number of TTA augmentations during inference