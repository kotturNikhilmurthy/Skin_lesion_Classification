import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from src.dataset import SkinDataset
from src.model import get_model
from src.config import CFG


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

    CFG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CFG.PLOTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved to {CFG.PLOTS_DIR / 'confusion_matrix.png'}")


def plot_roc_curves(y_true, y_probs, classes):
    """Plot ROC curves for each class"""
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))

    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each class
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(CFG.PLOTS_DIR / "roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 ROC curves saved to {CFG.PLOTS_DIR / 'roc_curves.png'}")


def advanced_tta(model, images):
    """
    Advanced Test-Time Augmentation with 8 transformations
    """
    batch_size = images.size(0)
    predictions = []

    with torch.no_grad():
        # 1. Original
        pred = model(images).softmax(1)
        predictions.append(pred)

        # 2. Horizontal flip
        pred = model(torch.flip(images, dims=[3])).softmax(1)
        predictions.append(pred)

        # 3. Vertical flip
        pred = model(torch.flip(images, dims=[2])).softmax(1)
        predictions.append(pred)

        # 4. Both flips
        pred = model(torch.flip(images, dims=[2, 3])).softmax(1)
        predictions.append(pred)

        # 5. Rotate 90
        pred = model(torch.rot90(images, k=1, dims=[2, 3])).softmax(1)
        predictions.append(pred)

        # 6. Rotate 180
        pred = model(torch.rot90(images, k=2, dims=[2, 3])).softmax(1)
        predictions.append(pred)

        # 7. Rotate 270
        pred = model(torch.rot90(images, k=3, dims=[2, 3])).softmax(1)
        predictions.append(pred)

        # 8. Transpose
        pred = model(torch.transpose(images, 2, 3)).softmax(1)
        predictions.append(pred)

    # Average all predictions
    final_pred = torch.stack(predictions).mean(dim=0)

    return final_pred


def main():
    print("\n" + "=" * 60)
    print("🔍 EVALUATION WITH ADVANCED TTA")
    print("=" * 60)
    print(f"[INFO] Device: {CFG.DEVICE}")

    # Load Data
    try:
        test_ds = SkinDataset(root_dir=CFG.TEST_DIR, train=False)
        print(f"📊 Test samples: {len(test_ds)}")
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    test_loader = DataLoader(
        test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY
    )

    # Load Model (try EMA first, then regular)
    model = get_model().to(CFG.DEVICE)

    # Try to load EMA model first
    ema_weights = CFG.MODEL_DIR / "best_model_ema.pth"
    regular_weights = CFG.MODEL_DIR / "best_model.pth"

    if ema_weights.exists():
        print("✅ Loading EMA model weights...")
        model.load_state_dict(torch.load(ema_weights, map_location=CFG.DEVICE))
    elif regular_weights.exists():
        print("✅ Loading regular model weights...")
        model.load_state_dict(torch.load(regular_weights, map_location=CFG.DEVICE))
    else:
        print("❌ No trained weights found!")
        return

    model.eval()

    y_true = []
    y_pred = []
    y_probs = []

    print("\n🔄 Running Advanced TTA (8 augmentations)...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(CFG.DEVICE, non_blocking=True)
            labels = labels.to(CFG.DEVICE, non_blocking=True)

            # Apply Advanced TTA
            probs = advanced_tta(model, images)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate metrics
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION REPORT")
    print("=" * 60)

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=CFG.CLASSES,
        digits=4
    )
    print(report)

    # Overall accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\n🎯 Overall Accuracy: {accuracy * 100:.2f}%")

    # Per-class accuracy
    print("\n📋 Per-Class Accuracy:")
    for i, class_name in enumerate(CFG.CLASSES):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_true[class_mask]).mean()
            print(f"   {class_name}: {class_acc * 100:.2f}%")

    # ROC-AUC scores
    if len(CFG.CLASSES) > 2:
        # Multi-class AUC
        y_true_bin = label_binarize(y_true, classes=range(len(CFG.CLASSES)))
        auc_scores = []
        for i, class_name in enumerate(CFG.CLASSES):
            auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            auc_scores.append(auc)
            print(f"   {class_name} AUC: {auc:.4f}")
        print(f"\n📊 Mean AUC: {np.mean(auc_scores):.4f}")

    # Save visualizations
    plot_confusion_matrix(y_true, y_pred, CFG.CLASSES)
    plot_roc_curves(y_true, y_probs, CFG.CLASSES)

    # Save results to file
    results = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "per_class_accuracy": {
            CFG.CLASSES[i]: float((y_pred[y_true == i] == y_true[y_true == i]).mean())
            for i in range(len(CFG.CLASSES))
            if (y_true == i).sum() > 0
        }
    }

    import json
    with open(CFG.PLOTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n💾 Results saved to {CFG.PLOTS_DIR / 'evaluation_results.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()