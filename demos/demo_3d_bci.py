"""Module 3 Test A: Adding Feature 3 for Word Generation.

Hypothesis: Word Generation was invisible to our 2D model because it lives
in the Left Language Cortex (Broca's area) in the Beta band (14-28 Hz).

We extract:
  Feature 1: Motor Alpha on Cz (8-13 Hz)  -> Motor Imagery
  Feature 2: Frontal Theta on Fz (4-8 Hz) -> Mental Subtraction
  Feature 3: Left Frontal Beta on FC5 (14-28 Hz) -> Word Generation
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
print("MODULE 3 TEST A: GIVING WORD GENERATION A VOICE (FEATURE 3)")
print("=" * 65)

# 1. Load training data (128 trials)
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

X_np = X_batch.cpu().numpy()
y_np = y_batch.cpu().numpy().astype(int)

ch_names = meta.get("ch_names", [])
cz_idx = ch_names.index("Cz") if "Cz" in ch_names else 0
fz_idx = ch_names.index("Fz") if "Fz" in ch_names else 1
fc5_idx = ch_names.index("FC5") if "FC5" in ch_names else 37

print(f"Extracted {len(y_np)} real trials.")
print(f"Selected Electrodes:")
print(f"  1. Motor Cortex:        {ch_names[cz_idx]} (Cz)")
print(f"  2. Frontal Working Mem: {ch_names[fz_idx]} (Fz)")
print(f"  3. Left Language Area:  {ch_names[fc5_idx]} (FC5 - Broca's area neighbor)")

# 2. Feature 1: Motor Alpha on Cz (8-13 Hz)
sos_alpha = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")
cz_filtered = sosfiltfilt(sos_alpha, X_np[:, cz_idx], axis=-1)
feat_1_alpha = np.log(np.mean(cz_filtered ** 2, axis=-1) + 1e-12)

# 3. Feature 2: Frontal Theta on Fz (4-8 Hz)
sos_theta = butter(4, [4.0, 8.0], btype="bandpass", fs=120.0, output="sos")
fz_filtered = sosfiltfilt(sos_theta, X_np[:, fz_idx], axis=-1)
feat_2_theta = np.log(np.mean(fz_filtered ** 2, axis=-1) + 1e-12)

# 4. Feature 3: Left Language Beta on FC5 (14-28 Hz)
sos_beta = butter(4, [14.0, 28.0], btype="bandpass", fs=120.0, output="sos")
fc5_filtered = sosfiltfilt(sos_beta, X_np[:, fc5_idx], axis=-1)
feat_3_beta = np.log(np.mean(fc5_filtered ** 2, axis=-1) + 1e-12)

# -----------------------------------------------------------------
# Compare 2D vs. 3D
# -----------------------------------------------------------------
scaler = StandardScaler()

# 2D Model (Cz Alpha + Fz Theta)
X_2d = scaler.fit_transform(np.column_stack([feat_1_alpha, feat_2_theta]))
clf_2d = RidgeClassifier(alpha=1.0)
scores_2d = cross_val_score(clf_2d, X_2d, y_np, cv=5, scoring="balanced_accuracy")

# 3D Model (Cz Alpha + Fz Theta + FC5 Beta)
X_3d = scaler.fit_transform(np.column_stack([feat_1_alpha, feat_2_theta, feat_3_beta]))
clf_3d = RidgeClassifier(alpha=1.0)
scores_3d = cross_val_score(clf_3d, X_3d, y_np, cv=5, scoring="balanced_accuracy")
clf_3d.fit(X_3d, y_np)

print("\n" + "=" * 65)
print("EXPERIMENTAL RESULTS:")
print(f"  Random Chance (3 classes):        33.33%")
print(f"  Old 2D Model (Cz + Fz):           {scores_2d.mean() * 100:.2f}%")
print(f"  New 3D Model (Cz + Fz + FC5 Beta): {scores_3d.mean() * 100:.2f}%  <-- Did it improve?")
print("=" * 65)

# 5. Generate 3D visualization plot
fig = plt.figure(figsize=(12, 9), dpi=150)
ax = fig.add_subplot(111, projection="3d")

class_info = [
    (0, "Motor Imagery ('mi')", "#d62728", "o"),
    (1, "Mental Subtraction ('sub')", "#1f77b4", "s"),
    (2, "Word Generation ('word')", "#2ca02c", "^"),
]

for cls_idx, cls_label, color, marker in class_info:
    mask = (y_np == cls_idx)
    ax.scatter(
        feat_1_alpha[mask],
        feat_2_theta[mask],
        feat_3_beta[mask],
        c=color,
        marker=marker,
        s=50,
        edgecolors="k",
        linewidths=0.5,
        alpha=0.85,
        label=f"Class {cls_idx}: {cls_label} ({mask.sum()} trials)",
    )

ax.set_title("Module 3 Test A: 3D Thought Space (Motor Cz x Frontal Fz x Language FC5)", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Feature 1: Motor Alpha (Cz, 8-13 Hz)", fontsize=9)
ax.set_ylabel("Feature 2: Frontal Theta (Fz, 4-8 Hz)", fontsize=9)
ax.set_zlabel("Feature 3: Language Beta (FC5, 14-28 Hz)", fontsize=9)
ax.legend(loc="upper left", frameon=True, fontsize=10)

plt.tight_layout()

out_path_track = track_dir / "toy_3d_bci.png"
out_path_brain = Path("/home/meghan/.gemini/antigravity/brain/5fc71298-bfc2-4c22-8082-f3a013e9ba88/toy_3d_bci.png")

plt.savefig(out_path_track, bbox_inches="tight")
plt.savefig(out_path_brain, bbox_inches="tight")
plt.close()

print(f"\n[OK] 3D Scatter plot saved to:")
print(f"     1. {out_path_track}")
print(f"     2. {out_path_brain}")
print("=" * 65)

