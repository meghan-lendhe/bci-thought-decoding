"""Module 2: Power and Logarithm in Action.

Demonstrates why squaring measures energy, and why taking np.log
transforms a skewed, outlier-heavy distribution into a clean bell curve.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("MODULE 2: POWER & LOGARITHM (THE BELL CURVE IN ACTION)")
print("=" * 65)

# 1. Load 64 trials
data_dir = track_dir / "data" / "neural_compet"
loaders, meta = load_task(
    "eeg",
    "_motor_imagery_stream",
    data_dir=data_dir,
    dataset=None,
    device="cpu",
    batch_size=64,
    seed=42,
    num_workers=0,
    subset="full",
    target_transform=lambda y: y.argmax(-1) if hasattr(y, "argmax") else y,
)

for X, y, info in loaders["train"]:
    break

# Extract Cz (Central Motor Cortex) across all 64 trials
ch_names = meta.get("ch_names", [])
cz_idx = ch_names.index("Cz") if "Cz" in ch_names else 0
raw_cz_all = X[:, cz_idx].cpu().numpy()  # shape (64, 480)

# 2. Filter all trials into Alpha (8-13 Hz)
sos = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")
filt_cz_all = sosfiltfilt(sos, raw_cz_all, axis=-1)

# 3. Compute Raw Power = Average of (x^2)
# (Notice how this is literally the variance of the wave!)
raw_power = np.mean(filt_cz_all ** 2, axis=-1)  # 64 numbers

# 4. Compute Log-Power = np.log(raw_power)
# (This is line 214 of master_ensemble.py!)
log_power = np.log(np.maximum(raw_power, 1e-12))

print(f"\n[A] Raw Power across 64 trials (in uV^2):")
print(f"    Min:  {raw_power.min():.2f} uV^2")
print(f"    Max:  {raw_power.max():.2f} uV^2")
print(f"    Mean: {raw_power.mean():.2f} uV^2")
print(f"    Notice how huge the gap is between min and max! Max is {raw_power.max()/raw_power.min():.1f}x larger!")

print(f"\n[B] Log-Power across 64 trials (np.log(raw_power)):")
print(f"    Min:  {log_power.min():.2f}")
print(f"    Max:  {log_power.max():.2f}")
print(f"    Mean: {log_power.mean():.2f}")
print(f"    The numbers are now evenly balanced around the mean!")

def text_histogram(values, title, n_bins=8):
    print(f"\n--- {title} ---")
    counts, bin_edges = np.histogram(values, bins=n_bins)
    max_count = max(counts)
    for i in range(n_bins):
        bar = "#" * int(counts[i] / max_count * 30)
        print(f"[{bin_edges[i]:6.1f} to {bin_edges[i+1]:6.1f}] | {bar} ({counts[i]})")

text_histogram(raw_power, "RAW POWER DISTRIBUTION (Skewed, Heavy Tail to the Right)")
text_histogram(log_power, "LOG-POWER DISTRIBUTION (Symmetric, Nice Bell Curve)")

print("\n" + "=" * 65)
print("OBSERVATION:")
print("1. In the Raw Power plot, notice how almost all trials pile up in the")
print("   first bin, with a few extreme numbers stretching way out to the right.")
print("2. In the Log-Power plot, notice how the bars form a symmetric bell curve,")
print("   peaking in the middle with gentle tails on both sides.")
print("   THIS is why linear models (Ridge) require np.log!")
print("=" * 65)

