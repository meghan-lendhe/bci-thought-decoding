"""Module 2: Fourier Brick-Wall Filter vs. Butterworth Filter.

Demonstrates:
1. The Gibbs Phenomenon / Ringing Trap (sharp frequency cutoff creates fake time ripples).
2. Circular Edge Smearing (FFT wraps the end of the window back into the beginning).
"""

import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("MODULE 2: WHY NOT FOURIER? (GIBBS RINGING & CIRCULAR WRAP-AROUND)")
print("=" * 65)

# 1. Load trial 1 (Cz)
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

ch_names = meta.get("ch_names", [])
cz_idx = ch_names.index("Cz") if "Cz" in ch_names else 0
raw_cz = X[0, cz_idx].cpu().numpy()  # 480 samples

# 2. Add a realistic sharp spike at sample 350 (e.g. an eye blink or visual feedback onset)
# To see how both filters respond to a sudden event!
spike_signal = raw_cz.copy()
spike_signal[350] += 50.0  # +50 uV sharp spike

# -------------------------------------------------------------
# FILTER 1: Ideal Fourier Brick-Wall Filter
# (FFT -> zero out everything outside 8-13 Hz -> IFFT)
# -------------------------------------------------------------
fft_coeffs = np.fft.rfft(spike_signal)
freqs = np.fft.rfftfreq(len(spike_signal), d=1.0/120.0)

# Brick-wall mask: 1.0 inside [8, 13] Hz, exactly 0.0 outside
mask = (freqs >= 8.0) & (freqs <= 13.0)
fourier_filtered = np.fft.irfft(fft_coeffs * mask, n=len(spike_signal))

# -------------------------------------------------------------
# FILTER 2: Butterworth 4th-Order Bandpass Filter (Smooth Roll-off)
# -------------------------------------------------------------
sos = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")
butter_filtered = sosfiltfilt(sos, spike_signal)

# -------------------------------------------------------------
# Let's inspect the artifacts!
# -------------------------------------------------------------

print("\n[A] Ringing around the Spike (Samples 330 to 370):")
print("    A sharp spike occurred at sample 350.")
print("    Let's see what each filter produces BEFORE and AFTER sample 350:")

def ascii_sparkline(data, width=50):
    min_v, max_v = data.min(), data.max()
    chars = " _.-~*^"
    norm = ((data - min_v) / (max_v - min_v + 1e-9) * (len(chars) - 1)).astype(int)
    # subsample or display
    step = max(1, len(data) // width)
    return "".join(chars[norm[i]] for i in range(0, len(data), step)[:width])

print("\n1. Butterworth 4th-Order (Smooth Roll-off):")
print(f"   Max amplitude near spike: {np.abs(butter_filtered[330:370]).max():.2f} uV")
print("   Wave: " + ascii_sparkline(butter_filtered[320:380]))
print("   Notice: Clean, well-damped response that quickly returns to baseline.")

print("\n2. Fourier Brick-Wall Filter (Sharp Cutoff):")
print(f"   Max amplitude near spike: {np.abs(fourier_filtered[330:370]).max():.2f} uV")
print("   Wave: " + ascii_sparkline(fourier_filtered[320:380]))
print("   Notice: Violent, artificial 'ringing' ripples that oscillate for a long time!")

print("\n" + "-" * 65)
print("[B] Circular Edge Wrap-Around (Samples 0 to 40 at the START of the window):")
print("    Remember: The spike occurred near the END of the window (sample 350).")
print("    Does the spike leak backward into the START of the window (sample 0-40)?")

start_butter = butter_filtered[:40]
start_fourier = fourier_filtered[:40]

print(f"\n1. Butterworth at the start (Samples 0-40):")
print(f"   Range: [{start_butter.min():.2f} uV to {start_butter.max():.2f} uV]")
print("   Wave: " + ascii_sparkline(start_butter))
print("   Notice: The spike at sample 350 has ZERO effect on sample 0.")

print(f"\n2. Fourier at the start (Samples 0-40):")
print(f"   Range: [{start_fourier.min():.2f} uV to {start_fourier.max():.2f} uV]")
print("   Wave: " + ascii_sparkline(start_fourier))
print("   Notice: Look at the edges! Because FFT assumes the signal loops in a circle,")
print("   ripples from the end of the trial WRAP AROUND and contaminate the beginning!")

print("\n" + "=" * 65)
print("TAKEAWAYS:")
print("1. A brick-wall cutoff in frequency creates sinc-function ripples in time (Gibbs ringing).")
print("2. FFT's circular assumption causes future events to leak into the past at the boundaries.")
print("3. Butterworth's smooth polynomial roll-off and time-domain equation prevent both!")
print("=" * 65)

