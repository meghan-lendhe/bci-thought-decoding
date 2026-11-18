"""Module 3: The Simplest 2D BCI Model (Just 2 Numbers).

Extracts only 2 human-interpretable neurophysiological features per trial:
  Feature 1: Log-Power of Alpha (8-13 Hz) at electrode Cz (Motor Cortex)
  Feature 2: Log-Power of Theta (4-8 Hz) at electrode Fz (Frontal Lobe)

Plots a real 2D scatter plot with decision boundaries and evaluates accuracy!
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("MODULE 3: CLASSIFYING THOUGHTS WITH JUST 2 NUMBERS")
print("=" * 65)

# 1. Load training data (2 batches = 128 trials)
data_dir = track_dir / "data" / "neural_compet"
loaders, meta = load_task(
    "eeg",
    "_motor_imagery_stream",
    data_dir=data_dir,
    dataset=None,
    device="cpu",
    batch_size=128,
    seed=42,
    num_workers=0,
    subset="full",
    target_transform=lambda y: y.argmax(-1) if hasattr(y, "argmax") else y,
)

for X_batch, y_batch, info in loaders["train"]:
    break

X_np = X_batch.cpu().numpy()  # (N, 41, 480)
y_np = y_batch.cpu().numpy().astype(int)  # (N,)

ch_names = meta.get("ch_names", [])
cz_idx = ch_names.index("Cz") if "Cz" in ch_names else 0
fz_idx = ch_names.index("Fz") if "Fz" in ch_names else 1

print(f"Extracted {len(y_np)} real trials.")
print(f"Using electrodes: {ch_names[cz_idx]} (Motor) and {ch_names[fz_idx]} (Frontal)")

# 2. Extract Feature 1: Motor Alpha Power at Cz (8-13 Hz)
sos_alpha = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")
cz_filtered = sosfiltfilt(sos_alpha, X_np[:, cz_idx], axis=-1)
feat_motor_alpha = np.log(np.mean(cz_filtered ** 2, axis=-1) + 1e-12)

# 3. Extract Feature 2: Frontal Theta Power at Fz (4-8 Hz)
sos_theta = butter(4, [4.0, 8.0], btype="bandpass", fs=120.0, output="sos")
fz_filtered = sosfiltfilt(sos_theta, X_np[:, fz_idx], axis=-1)
feat_frontal_theta = np.log(np.mean(fz_filtered ** 2, axis=-1) + 1e-12)

# Combine into a 2D feature matrix (N, 2)
X_2d = np.column_stack([feat_motor_alpha, feat_frontal_theta])

# 4. Standardize and Train a simple 2D Linear Model
scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

clf = RidgeClassifier(alpha=1.0)
cv_scores = cross_val_score(clf, X_2d_scaled, y_np, cv=5, scoring="balanced_accuracy")
clf.fit(X_2d_scaled, y_np)

print(f"\n[A] 2D Linear Classifier Performance (on just TWO numbers!):")
print(f"    Random Chance (3 classes): 33.33%")
print(f"    5-Fold Cross-Validation Balanced Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"    Training Set Accuracy: {clf.score(X_2d_scaled, y_np) * 100:.2f}%")

# 5. Generate Publication-Quality 2D Scatter Plot
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# Create a dense grid to show decision regions
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid_points)
Z = clf.predict(grid_scaled).reshape(xx.shape)

# Plot decision boundary regions
cmap_light = ["#ffebee", "#e3f2fd", "#e8f5e9"]  # light red, light blue, light green
ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=cmap_light, alpha=0.6)

# Plot real trials
class_info = [
    (0, "Motor Imagery ('mi')", "#d62728", "o"),      # Red circle
    (1, "Mental Subtraction ('sub')", "#1f77b4", "s"), # Blue square
    (2, "Word Generation ('word')", "#2ca02c", "^"),   # Green triangle
]

for cls_idx, cls_label, color, marker in class_info:
    mask = (y_np == cls_idx)
    ax.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        c=color,
        marker=marker,
        s=60,
        edgecolors="k",
        linewidths=0.7,
        alpha=0.85,
        label=f"Class {cls_idx}: {cls_label} ({mask.sum()} trials)",
    )

ax.set_title("Module 3: Can We Decode the Human Brain with Just 2 Numbers?", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Feature 1: Log-Power of Alpha (8-13 Hz) at Motor Cortex (Cz)\n<-- Low Power (ERD Activation) | High Power (Resting Idling) -->", fontsize=11)
ax.set_ylabel("Feature 2: Log-Power of Theta (4-8 Hz) at Frontal Lobe (Fz)\n<-- Low Working Memory | High Working Memory (Math Surge) -->", fontsize=11)
ax.legend(loc="upper left", frameon=True, fontsize=10)

plt.tight_layout()

# Save
out_path_track = track_dir / "toy_2d_bci.png"
out_path_brain = Path("/home/meghan/.gemini/antigravity/brain/5fc71298-bfc2-4c22-8082-f3a013e9ba88/toy_2d_bci.png")

plt.savefig(out_path_track, bbox_inches="tight")
plt.savefig(out_path_brain, bbox_inches="tight")
plt.close()

print(f"\n[OK] 2D Scatter plot saved to:")
print(f"     1. {out_path_track}")
print(f"     2. {out_path_brain}")
print("=" * 65)

