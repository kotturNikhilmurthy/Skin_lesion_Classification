import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from timm.utils import ModelEmaV2

from src.dataset import SkinDataset
from src.model import get_model
from src.config import CFG


# ==================================================
# ADVANCED LOSS FUNCTIONS
# ==================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================================================
# MIXUP & CUTMIX AUGMENTATIONS
# ==================================================
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=0.2):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Generate random box
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda based on actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================================================
# WARMUP SCHEDULER
# ==================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.lr_min + (self.base_lr - self.lr_min) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def save_plots(history):
    CFG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 5))

    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o', linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker='o', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training & Validation Loss", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color='green', marker='o', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Validation Accuracy", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Learning Rate
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["lr"], label="Learning Rate", color='orange', marker='o', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("LR", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CFG.PLOTS_DIR / "training_results.png", dpi=150)
    plt.close()
    print(f"📊 Plots saved to {CFG.PLOTS_DIR / 'training_results.png'}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"[INFO] Device: {CFG.DEVICE}")
    print(f"[INFO] Model: {CFG.MODEL_NAME}")
    print(f"[INFO] Image Size: {CFG.IMAGE_SIZE}")
    print(f"[INFO] Batch Size: {CFG.BATCH_SIZE}")

    # 1. Load Data
    try:
        train_ds = SkinDataset(root_dir=CFG.TRAIN_DIR, train=True)
    except RuntimeError as e:
        print(e)
        return

    if CFG.VAL_DIR.exists():
        val_ds = SkinDataset(root_dir=CFG.VAL_DIR, train=False)
    else:
        # Fallback split
        train_size = int(0.8 * len(train_ds))
        val_size = len(train_ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY, drop_last=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY
    )

    print(f"📊 Train samples: {len(train_ds)}")
    print(f"📊 Val samples: {len(val_ds)}")

    # 2. Model Setup
    model = get_model().to(CFG.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Total params: {total_params:,}")
    print(f"🔧 Trainable params: {trainable_params:,}")

    # EMA for stable predictions
    model_ema = ModelEmaV2(model, decay=0.9995, device=CFG.DEVICE)

    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.LR,
        weight_decay=CFG.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # Loss function with class weights
    class_weights = torch.tensor(CFG.CLASS_WEIGHTS, dtype=torch.float32).to(CFG.DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # Mixed precision training
    scaler = GradScaler()

    # Warmup + Cosine scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=CFG.WARMUP_EPOCHS,
        total_epochs=CFG.EPOCHS,
        lr_min=CFG.MIN_LR
    )

    CFG.MODEL_DIR.mkdir(exist_ok=True)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }
    best_val_acc = 0.0

    print("\n" + "=" * 60)
    print("🚀 Starting Advanced Training with Mixup/CutMix")
    print("=" * 60 + "\n")

    for epoch in range(1, CFG.EPOCHS + 1):
        # Update learning rate
        current_lr = scheduler.step(epoch - 1)

        # ==================== TRAINING ====================
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.EPOCHS}")

        for images, labels in loop:
            images = images.to(CFG.DEVICE, non_blocking=True)
            labels = labels.to(CFG.DEVICE, non_blocking=True)

            # Apply Mixup or CutMix randomly
            r = np.random.rand()
            if r < 0.33:
                # Mixup
                images, labels_a, labels_b, lam = mixup_data(images, labels, CFG.MIXUP_ALPHA)
                use_mixup = True
            elif r < 0.66:
                # CutMix
                images, labels_a, labels_b, lam = cutmix_data(images, labels, CFG.CUTMIX_ALPHA)
                use_mixup = True
            else:
                # No augmentation
                use_mixup = False

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(images)

                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()

            # Update EMA
            model_ema.update(model)

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=current_lr)

        train_loss /= len(train_loader)

        # ==================== VALIDATION ====================
        model_ema.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(CFG.DEVICE, non_blocking=True)
                labels = labels.to(CFG.DEVICE, non_blocking=True)

                # Use EMA model for validation
                outputs = model_ema.module(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"\n📈 Epoch {epoch}/{CFG.EPOCHS}")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Val Acc: {val_acc:.4f} ({val_acc * 100:.2f}%) | LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save both main model and EMA
            torch.save(model.state_dict(), CFG.MODEL_DIR / "best_model.pth")
            torch.save(model_ema.module.state_dict(), CFG.MODEL_DIR / "best_model_ema.pth")
            print(f"   ✅ NEW BEST! Saved at {best_val_acc * 100:.2f}%")

        print("-" * 60)

    # Save final model
    torch.save(model.state_dict(), CFG.MODEL_DIR / "final_model.pth")
    torch.save(model_ema.module.state_dict(), CFG.MODEL_DIR / "final_model_ema.pth")

    # Save training history
    with open(CFG.MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    save_plots(history)

    print("\n" + "=" * 60)
    print(f"🎉 Training Complete!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()