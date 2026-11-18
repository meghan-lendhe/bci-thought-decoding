"""Module 1: First-Principles Inspection of a Real EEG Trial.

Loads 1 real trial directly through neuralbench's task loader and inspects
its raw voltages, channel locations, time resolution, and waveform.
"""

import sys
from pathlib import Path
import numpy as np

# Add track directory to sys.path
track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("MODULE 1: FIRST-PRINCIPLES INSPECTION OF A REAL HUMAN EEG TRIAL")
print("=" * 65)

data_dir = track_dir / "data" / "neural_compet"

_PROTEUS_INTERFACES = {
    "AcquisitionBH": "BrainHero",
    "OnlineRawBH": "BrainHero",
    "AcquisitionGraz": "Graz",
    "OnlineRawGraz": "Graz",
}

def _proteus_interface(trigger):
    task = trigger.get("task")
    return _PROTEUS_INTERFACES.get(task, "Unknown")

def _proteus_run(trigger):
    return int(trigger.get("run", 1))

print("Loading 1 batch from pre-cached PROTEUS dataset...")
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
    context_of=_proteus_interface,
    order_of=_proteus_run,
)

loader = loaders["train"]

# Fetch the very first batch
for X, y, info in loader:
    break

# X has shape (B, 41, 480)
x0 = X[0].cpu().numpy() if hasattr(X[0], "cpu") else np.asarray(X[0])
y0 = y[0].cpu().numpy() if hasattr(y[0], "cpu") else np.asarray(y[0])

# Class mapping:
class_names = {
    0: "Motor Imagery ('mi')",
    1: "Mental Subtraction ('sub')",
    2: "Word Generation ('word')",
}
label = int(y0)

print(f"\n[1] The Raw Data Tensor:")
print(f"    Batch shape: {X.shape}")
print(f"    Single Trial shape: {x0.shape} -> (41 channels, 480 time points)")
print(f"    Trial Label: Class {label} -> {class_names.get(label, 'Unknown')}")
print(f"    Sampling Rate: {meta.get('sfreq', 120.0)} Hz -> 1 sample every {1000/120:.2f} milliseconds")
print(f"    Window Length: 480 samples / 120 Hz = {480/120:.1f} seconds total")

# Channel names:
ch_names = meta.get("ch_names", [f"Ch_{i}" for i in range(41)])

print(f"\n[2] The Scalp Electrodes (41 channels across the head):")
print(f"    Total channels: {len(ch_names)}")
print(f"    First 10 channels: {', '.join(ch_names[:10])}")

# Find Cz (motor cortex) and Fz (frontal lobe)
cz_idx = ch_names.index("Cz") if "Cz" in ch_names else 0
fz_idx = ch_names.index("Fz") if "Fz" in ch_names else 1

cz_wave = x0[cz_idx]
fz_wave = x0[fz_idx]

print(f"    Key locations: 'Cz' (index {cz_idx}, Central Motor Cortex), 'Fz' (index {fz_idx}, Frontal Midline)")

print(f"\n[3] Raw Voltage Values (in microvolts, uV):")
print(f"    Electrode {ch_names[cz_idx]} (Motor Cortex):")
print(f"    - First 10 consecutive time samples (spaced ~8.3 ms apart):")
print(f"      {np.round(cz_wave[:10], 2).tolist()}")
print(f"    - Full 4s range: [{cz_wave.min():.2f} uV to {cz_wave.max():.2f} uV]")
print(f"    - Mean: {cz_wave.mean():.2f} uV, Std: {cz_wave.std():.2f} uV")

print(f"\n    Electrode {ch_names[fz_idx]} (Frontal Lobe):")
print(f"    - Full 4s range: [{fz_wave.min():.2f} uV to {fz_wave.max():.2f} uV]")
print(f"    - Mean: {fz_wave.mean():.2f} uV, Std: {fz_wave.std():.2f} uV")

# ASCII wave visualization for Cz (first 60 samples = 0.5 second)
print(f"\n[4] ASCII Visualization of 0.5s of Brainwaves at Cz (60 samples):")
min_v, max_v = cz_wave[:60].min(), cz_wave[:60].max()
norm = ((cz_wave[:60] - min_v) / (max_v - min_v + 1e-9) * 15).astype(int)
grid = [[" " for _ in range(60)] for _ in range(16)]
for col, row in enumerate(norm):
    grid[15 - row][col] = "*"

for r in range(16):
    prefix = f"{max_v - r*(max_v-min_v)/15:6.1f} uV |" if r in (0, 7, 15) else "         |"
    print(prefix + "".join(grid[r]))
print("         +------------------------------------------------------------")
print("          0.0 s                                               0.5 s")

print("\n" + "=" * 65)
print("OBSERVATION QUESTIONS FOR DISCUSSION:")
print("1. Look at the amplitude: are the numbers in Volts or microvolts (~10 uV)?")
print("2. Look at the ASCII wave: how many ups and downs do you count in 0.5s?")
print("   (Hint: multiply that count by 2 to get the frequency in Hz!)")
print("=" * 65)
