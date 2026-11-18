"""Module 3 Test B: The Time Dynamics Hypothesis (Early vs. Late Slope).

Why did raw channel power fail?
Because Subject 1's baseline power is 10x larger than Subject 2's!
The classifier was grouping dots by SUBJECT, not by THOUGHT.

Hypothesis: In log space, Late - Early is the RELATIVE PERCENTAGE CHANGE:
  log(Late) - log(Early) = log(Late / Early)
This cancels out individual skull thickness!
And it captures the +31% working memory surge in Mental Subtraction.
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
print("MODULE 3 TEST B: TESTING THE TIME DIMENSION (DELTA POWER)")
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

X_np = X_batch.cpu().numpy()  # (128, 41, 480)
y_np = y_batch.cpu().numpy().astype(int)

ch_names = meta.get("ch_names", [])
cz_idx = ch_names.index("Cz") if "Cz" in ch_names else 0
fz_idx = ch_names.index("Fz") if "Fz" in ch_names else 1

# 2. Slice into Early Phase (0-240 samples, 1-3s) and Late Phase (240-480 samples, 3-5s)
# Remember: Slice FIRST before filtering to prevent boundary smearing!
split_idx = X_np.shape[-1] // 2
X_early = X_np[..., :split_idx]
X_late = X_np[..., split_idx:]

# Filter Cz (Alpha, 8-13 Hz)
sos_alpha = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")
cz_e = sosfiltfilt(sos_alpha, X_early[:, cz_idx], axis=-1)
cz_l = sosfiltfilt(sos_alpha, X_late[:, cz_idx], axis=-1)

p_cz_early = np.log(np.mean(cz_e ** 2, axis=-1) + 1e-12)
p_cz_late = np.log(np.mean(cz_l ** 2, axis=-1) + 1e-12)
delta_cz = p_cz_late - p_cz_early  # Slope of motor alpha

# Filter Fz (Theta, 4-8 Hz)
sos_theta = butter(4, [4.0, 8.0], btype="bandpass", fs=120.0, output="sos")
fz_e = sosfiltfilt(sos_theta, X_early[:, fz_idx], axis=-1)
fz_l = sosfiltfilt(sos_theta, X_late[:, fz_idx], axis=-1)

p_fz_early = np.log(np.mean(fz_e ** 2, axis=-1) + 1e-12)
p_fz_late = np.log(np.mean(fz_l ** 2, axis=-1) + 1e-12)
delta_fz = p_fz_late - p_fz_early  # Slope of frontal theta

# -------------------------------------------------------------
# Test Models
# -------------------------------------------------------------
scaler = StandardScaler()

# Model 1: Flat 4s Power (The failed 30% model)
sos_flat_a = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")
sos_flat_t = butter(4, [4.0, 8.0], btype="bandpass", fs=120.0, output="sos")
cz_flat = sosfiltfilt(sos_flat_a, X_np[:, cz_idx], axis=-1)
fz_flat = sosfiltfilt(sos_flat_t, X_np[:, fz_idx], axis=-1)
p_cz_flat = np.log(np.mean(cz_flat ** 2, axis=-1) + 1e-12)
p_fz_flat = np.log(np.mean(fz_flat ** 2, axis=-1) + 1e-12)

X_flat = scaler.fit_transform(np.column_stack([p_cz_flat, p_fz_flat]))
score_flat = cross_val_score(RidgeClassifier(alpha=1.0), X_flat, y_np, cv=5, scoring="balanced_accuracy").mean()

# Model 2: Early vs Late Features (4 features: early_cz, late_cz, early_fz, late_fz)
X_early_late = scaler.fit_transform(np.column_stack([p_cz_early, p_cz_late, p_fz_early, p_fz_late]))
score_early_late = cross_val_score(RidgeClassifier(alpha=1.0), X_early_late, y_np, cv=5, scoring="balanced_accuracy").mean()

# Model 3: Early/Late + Explicit Temporal Delta Slope (6 features)
X_with_delta = scaler.fit_transform(np.column_stack([p_cz_early, p_cz_late, delta_cz, p_fz_early, p_fz_late, delta_fz]))
score_with_delta = cross_val_score(RidgeClassifier(alpha=1.0), X_with_delta, y_np, cv=5, scoring="balanced_accuracy").mean()

print("\n" + "=" * 65)
print("EXPERIMENTAL RESULTS:")
print(f"  Random Chance (3 classes):                     33.33%")
print(f"  1. Flat 4s Average (No Time Dynamics):          {score_flat * 100:.2f}%")
print(f"  2. Sliced Early vs. Late (Doubled Resolution): {score_early_late * 100:.2f}%")
print(f"  3. Early + Late + Explicit Slope (Delta Power): {score_with_delta * 100:.2f}%")
print("=" * 65)

# 3. Plot the Temporal Slope: Delta Cz vs Delta Fz
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

class_info = [
    (0, "Motor Imagery ('mi')", "#d62728", "o"),
    (1, "Mental Subtraction ('sub')", "#1f77b4", "s"),
    (2, "Word Generation ('word')", "#2ca02c", "^"),
]

for cls_idx, cls_label, color, marker in class_info:
    mask = (y_np == cls_idx)
    ax.scatter(
        delta_cz[mask],
        delta_fz[mask],
        c=color,
        marker=marker,
        s=60,
        edgecolors="k",
        linewidths=0.6,
        alpha=0.85,
        label=f"Class {cls_idx}: {cls_label} (median Delta Fz = {np.median(delta_fz[mask]):+.2f})",
    )

ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
ax.set_title("Module 3 Test B: The Slope of Thought (Late Power - Early Power)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Slope of Motor Alpha at Cz: Delta = Log(Late) - Log(Early)\n<-- Alpha Dropped Over Time | Alpha Surged Over Time -->", fontsize=10)
ax.set_ylabel("Slope of Frontal Theta at Fz: Delta = Log(Late) - Log(Early)\n<-- Theta Dropped | Theta Surged (+31% Mental Math!) -->", fontsize=10)
ax.legend(loc="upper left", frameon=True, fontsize=10)

plt.tight_layout()

out_path_track = track_dir / "toy_temporal_slope.png"
out_path_brain = Path("/home/meghan/.gemini/antigravity/brain/5fc71298-bfc2-4c22-8082-f3a013e9ba88/toy_temporal_slope.png")

plt.savefig(out_path_track, bbox_inches="tight")
plt.savefig(out_path_brain, bbox_inches="tight")
plt.close()

print(f"\n[OK] Temporal Slope plot saved to:")
print(f"     1. {out_path_track}")
print(f"     2. {out_path_brain}")
print("=" * 65)

