"""Diagnostic Tool 1: Confusion Matrix & Class-by-Class Breakdown.

Inspects our Adaptive Master Ensemble on the test stream:
1. Which classes are performing well?
2. Which classes are confusing the model?
3. Where is the model making its mistakes?
"""

import sys
import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("DIAGNOSTIC TOOL 1: CONFUSION MATRIX & PER-CLASS ERROR ANALYSIS")
print("=" * 65)

# 1. Load the trained solver from outputs/AdaptiveMasterEnsemble
output_dir = track_dir / "outputs" / "AdaptiveMasterEnsemble"
sub_file = output_dir / "submission.py"

spec = importlib.util.spec_from_file_location("adaptive_sub", sub_file)
sub_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub_mod)

solver = sub_mod.Solver()
meta = {
    "submission_dir": output_dir,
    "n_chans": 41,
    "n_classes": 3,
    "sfreq": 120.0,
    "device": "cpu",
}

print("\n[1/3] Loading trained Adaptive Master Ensemble model...")
model = solver.load_model(meta)
model.reset_state()
print("      [OK] Model loaded with trained weights!")

# 2. Load the test stream
print("\n[2/3] Streaming test trials one-by-one...")
data_dir = track_dir / "data" / "neural_compet"
loaders, task_meta = load_task(
    "eeg",
    "_motor_imagery_stream",
    data_dir=data_dir,
    dataset=None,
    device="cpu",
    batch_size=1,
    seed=42,
    num_workers=0,
    subset="full",
    target_transform=lambda y: y.argmax(-1) if hasattr(y, "argmax") else y,
)

y_true_list = []
y_pred_list = []

for X_b, y_b, info in loaders["test"]:
    y_true_list.append(int(y_b.cpu().numpy()[0]))
    pred = model.predict(X_b)
    y_pred_list.append(int(pred[0]))

y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)

class_names = ["0: Motor Imagery", "1: Mental Subtraction", "2: Word Generation"]

# 3. Compute Metrics
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100  # Percentage per true class

print("\n" + "=" * 65)
print("CONFUSION MATRIX (RAW COUNTS):")
print("=" * 65)
print(f"{'True \\ Pred':<24} | {'MI (0)':<10} | {'Sub (1)':<10} | {'Word (2)':<10} | {'Total':<6}")
print("-" * 65)
for i, name in enumerate(class_names):
    print(f"{name:<24} | {cm[i, 0]:<10} | {cm[i, 1]:<10} | {cm[i, 2]:<10} | {cm[i].sum():<6}")

print("\n" + "=" * 65)
print("ACCURACY PER CLASS (PERCENTAGE OF CORRECT GUESSES):")
print("=" * 65)
for i, name in enumerate(class_names):
    print(f"• {name:<24}: {cm_norm[i, i]:5.1f}% correct ({cm[i, i]}/{cm[i].sum()} trials)")

# 4. Plot Heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
fig.suptitle("Diagnostic Analysis: Where is the Model Making Mistakes?", fontsize=14, fontweight="bold")

# Plot 1: Raw Counts
im1 = ax1.imshow(cm, cmap="Blues", interpolation="nearest")
ax1.set_title("1. Raw Trial Counts", fontsize=11, fontweight="bold", pad=10)
ax1.set_xticks(range(3))
ax1.set_yticks(range(3))
ax1.set_xticklabels(["MI (0)", "Sub (1)", "Word (2)"], fontsize=10)
ax1.set_yticklabels(["MI (0)", "Sub (1)", "Word (2)"], fontsize=10)
ax1.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
ax1.set_ylabel("True Label", fontsize=10, fontweight="bold")
for i in range(3):
    for j in range(3):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

# Plot 2: Percentage per True Class
im2 = ax2.imshow(cm_norm, cmap="Reds", vmin=0, vmax=100, interpolation="nearest")
ax2.set_title("2. Accuracy & Confusion Rate (%)", fontsize=11, fontweight="bold", pad=10)
ax2.set_xticks(range(3))
ax2.set_yticks(range(3))
ax2.set_xticklabels(["MI (0)", "Sub (1)", "Word (2)"], fontsize=10)
ax2.set_yticklabels(["MI (0)", "Sub (1)", "Word (2)"], fontsize=10)
ax2.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
ax2.set_ylabel("True Label", fontsize=10, fontweight="bold")
for i in range(3):
    for j in range(3):
        color = "white" if cm_norm[i, j] > 50 else "black"
        ax2.text(j, i, f"{cm_norm[i, j]:.1f}%", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Percentage (%)")
plt.tight_layout()

out_plot_track = track_dir / "confusion_matrix_diagnostic.png"
out_plot_brain = Path("/home/meghan/.gemini/antigravity/brain/5fc71298-bfc2-4c22-8082-f3a013e9ba88/confusion_matrix_diagnostic.png")
plt.savefig(out_plot_track, bbox_inches="tight")
plt.savefig(out_plot_brain, bbox_inches="tight")
plt.close()

print("\n" + "=" * 65)
print(f"[OK] Diagnostic heatmap saved to:")
print(f"     1. {out_plot_track}")
print(f"     2. {out_plot_brain}")
print("=" * 65)

