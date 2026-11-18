"""Module 2: Demystifying Filters.

Shows how a digital bandpass filter extracts a pure 10 Hz Alpha/Mu rhythm
from the messy, jagged raw EEG wave.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("MODULE 2: WHAT DOES A FILTER PHYSICALLY DO?")
print("=" * 65)

# 1. Load trial 1
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

for X, y, info in loaders["train"]:
    break

# Extract Cz (Central Motor Cortex)
ch_names = meta.get("ch_names", [])
cz_idx = ch_names.index("Cz") if "Cz" in ch_names else 0
raw_cz = X[0, cz_idx].cpu().numpy()

# 2. Design the Alpha Bandpass Filter (8.0 to 13.0 Hz)
# This is the exact function used in line 173 of master_ensemble.py!
sfreq = 120.0
f_low, f_high = 8.0, 13.0
sos = butter(4, [f_low, f_high], btype="bandpass", fs=sfreq, output="sos")
filtered_cz = sosfiltfilt(sos, raw_cz)

# Look at 0.5 seconds (60 samples)
raw_segment = raw_cz[:60]
filt_segment = filtered_cz[:60]

print("\n[A] What is a Digital Filter Mathematically?")
print("    It is literally a weighted sliding average of nearby samples.")
print("    - A Lowpass filter adds neighbors -> smooths away fast jitters.")
print("    - A Highpass filter subtracts neighbors -> cancels slow drifts.")
print("    - A Bandpass filter does both -> mutes everything EXCEPT 8-13 Hz!")

print("\n[B] Side-by-Side Comparison of 0.5 seconds (60 samples):")
print("\n--- 1. RAW WAVE (The messy, jagged combination of all brain rhythms) ---")
min_r, max_r = raw_segment.min(), raw_segment.max()
norm_r = ((raw_segment - min_r) / (max_r - min_r + 1e-9) * 10).astype(int)
for r in range(11):
    line = "".join("*" if norm_r[c] == (10 - r) else " " for c in range(60))
    prefix = f"{max_r - r*(max_r-min_r)/10:5.1f} uV |" if r in (0, 5, 10) else "        |"
    print(prefix + line)
print("        +------------------------------------------------------------")
print("         0.0 s                                               0.5 s")

print("\n--- 2. FILTERED WAVE (8-13 Hz Alpha/Mu rhythm isolated) ---")
min_f, max_f = filt_segment.min(), filt_segment.max()
norm_f = ((filt_segment - min_f) / (max_f - min_f + 1e-9) * 10).astype(int)
for r in range(11):
    line = "".join("*" if norm_f[c] == (10 - r) else " " for c in range(60))
    prefix = f"{max_f - r*(max_f-min_f)/10:5.1f} uV |" if r in (0, 5, 10) else "        |"
    print(prefix + line)
print("        +------------------------------------------------------------")
print("         0.0 s                                               0.5 s")

print("\n" + "=" * 65)
print("OBSERVATION:")
print("Count the peaks in the filtered wave now!")
print("In 0.5 seconds, you will see roughly 5 distinct, smooth waves.")
print("5 peaks in 0.5s = 10 peaks in 1.0s = exactly 10 Hz (Alpha/Mu rhythm)!")
print("=" * 65)

