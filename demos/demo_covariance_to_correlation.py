"""Module 4: Demystifying Covariance & Correlation (41 Channels).

Visualizes:
1. What a 41x41 Covariance Matrix looks like (Power on diagonal, Synchrony on off-diagonal).
2. Why line 178 of master_ensemble.py converts Covariance to Correlation
   (_cov_to_corr) to eliminate human skull thickness disparities!
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pyriemann.estimation import Covariances
from scipy.signal import butter, sosfiltfilt

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("MODULE 4: VISUALIZING THE 41x41 COVARIANCE & CORRELATION MANIFOLD")
print("=" * 65)

# 1. Load Trial 1
data_dir = track_dir / "data" / "neural_compet"
loaders, meta = load_task(
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

for X_batch, y_batch, info in loaders["train"]:
    break

X_trial = X_batch[0].cpu().numpy()  # (41, 480)
ch_names = meta.get("ch_names", [f"Ch_{i}" for i in range(41)])

# 2. Filter in Alpha band (8-13 Hz)
sos = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")
X_filt = sosfiltfilt(sos, X_trial, axis=-1)  # (41, 480)

# 3. Compute Ledoit-Wolf Shrinkage Covariance (41 x 41)
# (This is line 198 of master_ensemble.py!)
cov_estimator = Covariances(estimator="lwf")
cov_matrix = cov_estimator.fit_transform(X_filt[np.newaxis, ...])[0]  # (41, 41)

# 4. Convert Covariance to Correlation using the EXACT function from line 178 of master_ensemble.py!
def _cov_to_corr(covs):
    diags = np.diagonal(covs, axis1=-2, axis2=-1)
    stds = np.sqrt(np.maximum(diags, 1e-12))[..., np.newaxis]
    corrs = covs / (stds @ stds.swapaxes(-2, -1))
    for i in range(len(corrs)):
        np.fill_diagonal(corrs[i], 1.0)
    return corrs, diags

corr_matrix, diags = _cov_to_corr(cov_matrix[np.newaxis, ...])
corr_matrix = corr_matrix[0]
diags = diags[0]

print(f"\n[A] Covariance Matrix (41 x 41):")
print(f"    Diagonal (Individual Electrode Power):")
print(f"      Min power: {diags.min():.2f} uV^2, Max power: {diags.max():.2f} uV^2")
print(f"    Off-Diagonal (Scale x Synchrony):")
print(f"      Range: [{cov_matrix.min():.2f} to {cov_matrix.max():.2f}]")
print(f"      Notice: The numbers are in raw microvolts squared (uV^2).")
print(f"      A person with a thick skull will have small numbers;")
print(f"      A person with a thin skull will have huge numbers!")

print(f"\n[B] Correlation Matrix (41 x 41):")
print(f"    Diagonal: Exactly {corr_matrix.diagonal().min():.1f} to {corr_matrix.diagonal().max():.1f} for every channel!")
print(f"    Off-Diagonal: Strictly bounded between [{corr_matrix.min():.2f} and {corr_matrix.max():.2f}]")
print(f"    The skull thickness scale has been 100% ELIMINATED.")
print(f"    Every human on earth now shares the exact same scale: [-1.0, +1.0]!")

# 5. Plot Side-by-Side Heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
fig.suptitle("Module 4: Spatial Geometry — Covariance vs. Scale-Invariant Correlation", fontsize=15, fontweight="bold", y=0.98)

# Heatmap 1: Raw Covariance
im1 = ax1.imshow(cov_matrix, cmap="viridis", interpolation="nearest")
ax1.set_title(f"1. Raw Covariance (41 x 41)\nMixed with skull voltage scale (0 to {cov_matrix.max():.1f} uV^2)", fontsize=11, fontweight="bold", pad=10)
ax1.set_xlabel("Electrode Index (0 to 40)", fontsize=10)
ax1.set_ylabel("Electrode Index (0 to 40)", fontsize=10)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="uV^2")

# Heatmap 2: Normalized Correlation
im2 = ax2.imshow(corr_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
ax2.set_title("2. Correlation (41 x 41) — Line 178 of Our Model\nPure Network Synchrony: Skull scale removed! ([-1 to +1])", fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Electrode Index (0 to 40)", fontsize=10)
ax2.set_ylabel("Electrode Index (0 to 40)", fontsize=10)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Correlation (-1 to +1)")

plt.tight_layout()

out_path_track = track_dir / "cov_vs_corr.png"
out_path_brain = Path("/home/meghan/.gemini/antigravity/brain/5fc71298-bfc2-4c22-8082-f3a013e9ba88/cov_vs_corr.png")

plt.savefig(out_path_track, bbox_inches="tight")
plt.savefig(out_path_brain, bbox_inches="tight")
plt.close()

print(f"\n[OK] Heatmap plot saved to:")
print(f"     1. {out_path_track}")
print(f"     2. {out_path_brain}")
print("=" * 65)

