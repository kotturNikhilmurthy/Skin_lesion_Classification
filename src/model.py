import timm
import torch.nn as nn
from .config import CFG


def get_model():
    """
    Creates and returns a powerful pre-trained model
    configured for skin lesion classification.

    Uses EfficientNetV2-M by default - excellent balance of
    accuracy and efficiency for medical imaging.
    """
    model = timm.create_model(
        model_name=CFG.MODEL_NAME,
        pretrained=True,
        num_classes=CFG.NUM_CLASSES,
        drop_rate=0.3,  # Dropout for regularization
        drop_path_rate=0.2  # Stochastic depth
    )

    return model


def get_ensemble_models():
    """
    For even higher accuracy, create an ensemble of models.
    Use this during inference for best results.
    """
    models = []

    # Model 1: EfficientNetV2-M
    model1 = timm.create_model(
        "tf_efficientnetv2_m",
        pretrained=True,
        num_classes=CFG.NUM_CLASSES,
        drop_rate=0.3
    )
    models.append(model1)

    # Model 2: ConvNeXt Base
    model2 = timm.create_model(
        "convnext_base",
        pretrained=True,
        num_classes=CFG.NUM_CLASSES,
        drop_rate=0.3
    )
    models.append(model2)

    # Model 3: Swin Transformer
    model3 = timm.create_model(
        "swin_base_patch4_window12_384",
        pretrained=True,
        num_classes=CFG.NUM_CLASSES,
        drop_rate=0.3
    )
    models.append(model3)

    return models