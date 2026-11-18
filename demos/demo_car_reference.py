"""Module 4: What if we remove the common reference? (Common Average Referencing - CAR).

Shows how subtracting the spatial mean across all 41 electrodes removes the
common elevator, revealing true negative dipole correlations!
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
print("MODULE 4: WHAT IF WE REMOVE THE COMMON REFERENCE (CAR)?")
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

X_raw = X_batch[0].cpu().numpy()  # (41, 480)

# 2. Method 1: Original Earlobe Reference
sos = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")
X_filt_raw = sosfiltfilt(sos, X_raw, axis=-1)

# Method 2: Common Average Reference (CAR)
# At every millisecond, compute the average voltage of all 41 electrodes, and subtract it!
spatial_mean = np.mean(X_raw, axis=0, keepdims=True)  # (1, 480)
X_car = X_raw - spatial_mean  # (41, 480)
X_filt_car = sosfiltfilt(sos, X_car, axis=-1)

# 3. Compute Covariance and Correlation for both
def _cov_to_corr(covs):
    diags = np.diagonal(covs, axis1=-2, axis2=-1)
    stds = np.sqrt(np.maximum(diags, 1e-12))[..., np.newaxis]
    corrs = covs / (stds @ stds.swapaxes(-2, -1))
    for i in range(len(corrs)):
        np.fill_diagonal(corrs[i], 1.0)
    return corrs

cov_est = Covariances(estimator="lwf")
cov_raw = cov_est.fit_transform(X_filt_raw[np.newaxis, ...])
corr_raw = _cov_to_corr(cov_raw)[0]

cov_car = cov_est.fit_transform(X_filt_car[np.newaxis, ...])
corr_car = _cov_to_corr(cov_car)[0]

print(f"\n[A] Original Earlobe Reference Correlation:")
print(f"    Min: {corr_raw.min():.2f}, Max: {corr_raw.max():.2f}")
print(f"    Almost all values are positive because of the shared reference wire!")

print(f"\n[B] Common Average Reference (CAR) Correlation:")
print(f"    Min: {corr_car.min():.2f}, Max: {corr_car.max():.2f}")
print(f"    Look at that! Negative values have appeared! Range is now [{corr_car.min():.2f} to {corr_car.max():.2f}]")

# 4. Plot Side-by-Side Heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
fig.suptitle("What Happens When We Remove the Common Reference (CAR)?", fontsize=15, fontweight="bold", y=0.98)

im1 = ax1.imshow(corr_raw, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
ax1.set_title("1. Original Earlobe Reference\nShared reference wire shifts all correlations positive (+0.2 to +1.0)", fontsize=11, fontweight="bold", pad=10)
ax1.set_xlabel("Electrode Index (0 to 40)", fontsize=10)
ax1.set_ylabel("Electrode Index (0 to 40)", fontsize=10)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Correlation (-1 to +1)")

im2 = ax2.imshow(corr_car, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
ax2.set_title(f"2. Common Average Reference (CAR)\nCommon reference removed! True negative dipoles appear ({corr_car.min():.2f} to +1.0)", fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Electrode Index (0 to 40)", fontsize=10)
ax2.set_ylabel("Electrode Index (0 to 40)", fontsize=10)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Correlation (-1 to +1)")

plt.tight_layout()

out_path_track = track_dir / "car_reference_comparison.png"
out_path_brain = Path("/home/meghan/.gemini/antigravity/brain/5fc71298-bfc2-4c22-8082-f3a013e9ba88/car_reference_comparison.png")

plt.savefig(out_path_track, bbox_inches="tight")
plt.savefig(out_path_brain, bbox_inches="tight")
plt.close()

print(f"\n[OK] Comparison plot saved to:")
print(f"     1. {out_path_track}")
print(f"     2. {out_path_brain}")
print("=" * 65)

