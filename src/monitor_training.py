"""
Real-time Training Monitor
Run this in a separate terminal while training to see live updates
"""
import json
import time
from pathlib import Path
import os


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def monitor_training(history_path, refresh_interval=5):
    """
    Monitor training progress in real-time

    Args:
        history_path: Path to training_history.json
        refresh_interval: Seconds between updates
    """
    print("🔍 Training Monitor Started")
    print("Press Ctrl+C to stop\n")

    last_epoch = 0

    try:
        while True:
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)

                if len(history['train_loss']) > last_epoch:
                    last_epoch = len(history['train_loss'])

                    clear_screen()

                    print("=" * 70)
                    print("🚀 REAL-TIME TRAINING MONITOR")
                    print("=" * 70)
                    print(f"Monitoring: {history_path}")
                    print(f"Last updated: {time.strftime('%H:%M:%S')}")
                    print("=" * 70)

                    # Current stats
                    current_epoch = len(history['train_loss'])
                    train_loss = history['train_loss'][-1]
                    val_loss = history['val_loss'][-1]
                    val_acc = history['val_acc'][-1] * 100

                    # Best stats
                    best_val_acc = max(history['val_acc']) * 100
                    best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1

                    print(f"\n📊 CURRENT STATUS (Epoch {current_epoch})")
                    print("-" * 70)
                    print(f"  Train Loss:      {train_loss:.4f}")
                    print(f"  Val Loss:        {val_loss:.4f}")
                    print(f"  Val Accuracy:    {val_acc:.2f}%")

                    if 'lr' in history and len(history['lr']) > 0:
                        current_lr = history['lr'][-1]
                        print(f"  Learning Rate:   {current_lr:.6f}")

                    print(f"\n🏆 BEST PERFORMANCE")
                    print("-" * 70)
                    print(f"  Best Val Acc:    {best_val_acc:.2f}%")
                    print(f"  Best Epoch:      {best_epoch}")

                    # Progress bar
                    from src.config import CFG
                    total_epochs = CFG.EPOCHS
                    progress = current_epoch / total_epochs
                    bar_length = 50
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)

                    print(f"\n⏳ PROGRESS")
                    print("-" * 70)
                    print(f"  [{bar}] {progress * 100:.1f}%")
                    print(f"  Epoch {current_epoch}/{total_epochs}")

                    # Last 5 epochs
                    if current_epoch >= 5:
                        print(f"\n📈 RECENT HISTORY (Last 5 Epochs)")
                        print("-" * 70)
                        print("  Epoch | Train Loss | Val Loss | Val Acc")
                        print("-" * 70)
                        for i in range(max(0, current_epoch - 5), current_epoch):
                            e = i + 1
                            tl = history['train_loss'][i]
                            vl = history['val_loss'][i]
                            va = history['val_acc'][i] * 100
                            marker = "  👑" if va == best_val_acc else ""
                            print(f"  {e:5d} | {tl:10.4f} | {vl:8.4f} | {va:6.2f}%{marker}")

                    # Overfitting check
                    loss_gap = val_loss - train_loss
                    print(f"\n🔍 ANALYSIS")
                    print("-" * 70)
                    print(f"  Loss Gap (Val - Train): {loss_gap:.4f}")

                    if loss_gap > 0.5:
                        print("  ⚠️  WARNING: Significant overfitting detected!")
                        print("      Consider: early stopping, more regularization")
                    elif loss_gap > 0.2:
                        print("  ℹ️  Slight overfitting (normal for complex models)")
                    else:
                        print("  ✅ Good generalization")

                    # Predictions
                    if current_epoch >= 10:
                        recent_accs = history['val_acc'][-5:]
                        trend = (recent_accs[-1] - recent_accs[0]) / 5 * 100

                        print(
                            f"\n  Recent trend: {'📈 Improving' if trend > 0 else '📉 Declining'} ({trend:+.2f}% per epoch)")

                        if best_val_acc >= 90:
                            print("\n  🎉 TARGET ACHIEVED! Accuracy ≥ 90%")
                        elif val_acc > 85:
                            print(f"\n  💪 Getting close! Need {90 - val_acc:.1f}% more")

                    print("\n" + "=" * 70)
                    print(f"Next update in {refresh_interval} seconds... (Ctrl+C to stop)")
                    print("=" * 70)

            else:
                print(f"⏳ Waiting for training to start...")
                print(f"   Looking for: {history_path}")

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user")
        print(f"Final Stats: Epoch {current_epoch}, Val Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    from src.config import CFG

    history_path = CFG.MODEL_DIR / "training_history.json"
    monitor_training(history_path, refresh_interval=5)