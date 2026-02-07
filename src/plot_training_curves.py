import json
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import CFG


def main():
    history_path = CFG.MODEL_DIR / "training_history.json"

    if not history_path.exists():
        raise FileNotFoundError(
            "training_history.json not found. Train the model first."
        )

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Analysis', fontsize=16, fontweight='bold')

    # =========================
    # SUBPLOT 1: Loss Curves
    # =========================
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["train_loss"], label="Train Loss",
             marker='o', linewidth=2, markersize=4)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss",
             marker='s', linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training vs Validation Loss", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # =========================
    # SUBPLOT 2: Validation Accuracy
    # =========================
    ax2 = axes[0, 1]
    ax2.plot(epochs, [acc * 100 for acc in history["val_acc"]],
             label="Validation Accuracy", color='green',
             marker='o', linewidth=2, markersize=4)
    ax2.axhline(y=90, color='r', linestyle='--', linewidth=2, label='90% Target')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Validation Accuracy", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add best accuracy annotation
    best_acc = max(history["val_acc"]) * 100
    best_epoch = history["val_acc"].index(max(history["val_acc"])) + 1
    ax2.annotate(f'Best: {best_acc:.2f}%\nEpoch {best_epoch}',
                 xy=(best_epoch, best_acc),
                 xytext=(best_epoch + 5, best_acc - 5),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=10, fontweight='bold', color='red')

    # =========================
    # SUBPLOT 3: Learning Rate
    # =========================
    ax3 = axes[1, 0]
    if "lr" in history and len(history["lr"]) > 0:
        ax3.plot(epochs, history["lr"], label="Learning Rate",
                 color='orange', marker='o', linewidth=2, markersize=4)
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel("Learning Rate", fontsize=12)
        ax3.set_title("Learning Rate Schedule", fontsize=13, fontweight='bold')
        ax3.set_yscale('log')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No LR data available',
                 ha='center', va='center', fontsize=12)
        ax3.set_title("Learning Rate Schedule", fontsize=13, fontweight='bold')

    # =========================
    # SUBPLOT 4: Training Stats
    # =========================
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate statistics
    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    final_val_acc = history["val_acc"][-1] * 100
    best_val_acc = max(history["val_acc"]) * 100
    best_epoch = history["val_acc"].index(max(history["val_acc"])) + 1

    # Overfitting check
    loss_gap = final_val_loss - final_train_loss
    if loss_gap > 0.5:
        overfitting_status = "⚠️ Possible overfitting"
    elif loss_gap > 0.2:
        overfitting_status = "✓ Slight overfitting"
    else:
        overfitting_status = "✓ Good generalization"

    stats_text = f"""
    TRAINING SUMMARY
    {'=' * 40}

    Final Training Loss:      {final_train_loss:.4f}
    Final Validation Loss:    {final_val_loss:.4f}
    Loss Gap:                 {loss_gap:.4f}

    Final Validation Acc:     {final_val_acc:.2f}%
    Best Validation Acc:      {best_val_acc:.2f}%
    Best Epoch:               {best_epoch}

    Total Epochs:             {len(epochs)}

    Status:                   {overfitting_status}

    {'=' * 40}
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save figure
    CFG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CFG.PLOTS_DIR / "training_curves_detailed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Detailed training curves saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()