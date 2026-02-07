"""
Quick Start Script for Skin Lesion Classification
Run this to execute the full pipeline
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 70)
    print(f"🚀 {description}")
    print("=" * 70)

    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"✅ {description} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        return False


def main():
    print("\n" + "🎯" * 35)
    print("    SKIN LESION CLASSIFICATION - QUICK START")
    print("🎯" * 35)

    # Check if we're in the right directory
    if not Path("src/config.py").exists():
        print("\n❌ ERROR: Please run this script from the project root directory")
        print("   (The directory containing the 'src' folder)")
        sys.exit(1)

    print("\n📋 Pipeline Steps:")
    print("   1. Train the model")
    print("   2. Evaluate on test set")
    print("   3. Generate visualizations")
    print("   4. Show results")

    response = input("\n❓ Start training? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("❌ Cancelled by user")
        sys.exit(0)

    # Step 1: Training
    success = run_command(
        f"{sys.executable} -m src.train",
        "STEP 1/3: Training Model"
    )

    if not success:
        print("\n❌ Training failed! Please check the error messages above.")
        sys.exit(1)

    # Step 2: Evaluation
    success = run_command(
        f"{sys.executable} -m src.evaluate",
        "STEP 2/3: Evaluating Model"
    )

    if not success:
        print("\n⚠️  Evaluation failed, but model was trained successfully")
        print("   You can run evaluation manually later with:")
        print(f"   {sys.executable} -m src.evaluate")

    # Step 3: Visualizations
    success = run_command(
        f"{sys.executable} -m src.plot_training_curves",
        "STEP 3/3: Generating Visualizations"
    )

    if not success:
        print("\n⚠️  Visualization generation failed")
        print("   You can run it manually later with:")
        print(f"   {sys.executable} -m src.plot_training_curves")

    # Show results
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETED!")
    print("=" * 70)

    # Try to load and display results
    try:
        import json
        from src.config import CFG

        # Training history
        history_path = CFG.MODEL_DIR / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)

            best_val_acc = max(history['val_acc']) * 100
            final_val_acc = history['val_acc'][-1] * 100

            print(f"\n📊 TRAINING RESULTS:")
            print(f"   Best Validation Accuracy:  {best_val_acc:.2f}%")
            print(f"   Final Validation Accuracy: {final_val_acc:.2f}%")
            print(f"   Total Epochs Completed:    {len(history['train_loss'])}")

            if best_val_acc >= 90:
                print("\n   🎉🎉🎉 TARGET ACHIEVED! Accuracy ≥ 90% 🎉🎉🎉")
            elif best_val_acc >= 85:
                print(f"\n   💪 Close! Just {90 - best_val_acc:.1f}% away from target")
            else:
                print(f"\n   📈 Need {90 - best_val_acc:.1f}% more to reach 90% target")

        # Evaluation results
        eval_results_path = CFG.PLOTS_DIR / "evaluation_results.json"
        if eval_results_path.exists():
            with open(eval_results_path, 'r') as f:
                results = json.load(f)

            test_acc = results['accuracy'] * 100
            print(f"\n📊 TEST SET RESULTS:")
            print(f"   Test Accuracy: {test_acc:.2f}%")
            print("\n   Per-Class Accuracy:")
            for class_name, acc in results['per_class_accuracy'].items():
                print(f"      {class_name:25s}: {acc * 100:5.2f}%")

        print(f"\n📁 OUTPUT FILES:")
        print(f"   Models:         {CFG.MODEL_DIR}")
        print(f"   Visualizations: {CFG.PLOTS_DIR}")

        print("\n📸 Generated Files:")
        if (CFG.PLOTS_DIR / "training_results.png").exists():
            print(f"   ✅ Training curves:  {CFG.PLOTS_DIR / 'training_results.png'}")
        if (CFG.PLOTS_DIR / "confusion_matrix.png").exists():
            print(f"   ✅ Confusion matrix: {CFG.PLOTS_DIR / 'confusion_matrix.png'}")
        if (CFG.PLOTS_DIR / "roc_curves.png").exists():
            print(f"   ✅ ROC curves:       {CFG.PLOTS_DIR / 'roc_curves.png'}")

    except Exception as e:
        print(f"\n⚠️  Could not load results: {e}")

    print("\n" + "=" * 70)
    print("📝 NEXT STEPS:")
    print("=" * 70)
    print("1. Check the visualizations in the 'reports' folder")
    print("2. Review the confusion matrix for per-class performance")
    print("3. Use the trained model for inference:")
    print(f"   {sys.executable} -m src.inference path/to/image.jpg")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()