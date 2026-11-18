"""Module 2 Plotting: Fourier Brick-Wall vs. Butterworth Filter.

Generates a publication-quality high-resolution figure clearly visualizing:
1. Full 4-second time series.
2. The Gibbs Phenomenon (violent ringing ripples around a sudden spike).
3. Circular Edge Wrap-Around (FFT periodization leaking future spikes into the past).
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("GENERATING HIGH-RESOLUTION PLOT: FOURIER VS. BUTTERWORTH")
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
raw_cz = X[0, cz_idx].cpu().numpy()  # 480 samples, 120 Hz = 4.0 seconds
time_axis = np.arange(len(raw_cz)) / 120.0  # seconds

# 2. Add realistic spike at sample 350 (t = 2.92s)
spike_signal = raw_cz.copy()
spike_sample = 350
spike_time = spike_sample / 120.0
spike_signal[spike_sample] += 60.0  # +60 uV sharp event (e.g. eye blink / cue)

# 3. Filter 1: Ideal Fourier Brick-Wall Filter (8 - 13 Hz)
fft_coeffs = np.fft.rfft(spike_signal)
freqs = np.fft.rfftfreq(len(spike_signal), d=1.0 / 120.0)
mask = (freqs >= 8.0) & (freqs <= 13.0)
fourier_filtered = np.fft.irfft(fft_coeffs * mask, n=len(spike_signal))

# 4. Filter 2: Butterworth 4th-Order Bandpass (8 - 13 Hz)
sos = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")
butter_filtered = sosfiltfilt(sos, spike_signal)

# 5. Create 3-panel figure
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=150)
fig.suptitle("Why We Don't Use Fourier (FFT) Brick-Wall Filters in EEG", fontsize=16, fontweight="bold", y=0.98)

# Panel 1: Full 4-Second Trial
ax1 = axes[0]
ax1.plot(time_axis, spike_signal, color="gray", alpha=0.5, linewidth=1.0, label="Raw Signal + Spike at 2.92s")
ax1.plot(time_axis, butter_filtered, color="#1f77b4", linewidth=1.5, label="Butterworth 4th-Order (Smooth Roll-off)")
ax1.plot(time_axis, fourier_filtered, color="#d62728", linewidth=1.2, linestyle="--", alpha=0.85, label="Fourier Brick-Wall (Sharp Cutoff)")
ax1.axvline(spike_time, color="orange", linestyle=":", linewidth=1.5, label="Spike Event (Sample 350)")
ax1.set_title("1. Full 4.0-Second Window (480 samples @ 120 Hz)", fontsize=12, fontweight="bold", pad=8)
ax1.set_xlabel("Time (seconds)", fontsize=10)
ax1.set_ylabel("Amplitude (uV)", fontsize=10)
ax1.set_xlim(0, 4.0)
ax1.legend(loc="upper right", frameon=True, fontsize=9)

# Panel 2: Zoom on Spike (The Gibbs Ringing Trap)
ax2 = axes[1]
zoom_spike = slice(310, 390)
t_zoom = time_axis[zoom_spike]
ax2.plot(t_zoom, spike_signal[zoom_spike], color="black", linewidth=1.2, label="Raw Signal with Spike")
ax2.plot(t_zoom, butter_filtered[zoom_spike], color="#1f77b4", linewidth=2.2, label="Butterworth: Clean, Damped Response")
ax2.plot(t_zoom, fourier_filtered[zoom_spike], color="#d62728", linewidth=2.0, linestyle="-", label="Fourier: Violent Sinc Ringing Ripples!")
ax2.axvline(spike_time, color="orange", linestyle=":", linewidth=2.0)
ax2.set_title("2. Zoom-in: The Gibbs Phenomenon (Ringing Trap Around Sudden Events)", fontsize=12, fontweight="bold", pad=8)
ax2.set_xlabel("Time (seconds)", fontsize=10)
ax2.set_ylabel("Amplitude (uV)", fontsize=10)
ax2.annotate(
    "Sharp cliff in frequency produces\nendless fake ripples in time!\n(Gibbs Phenomenon)",
    xy=(spike_time + 0.12, fourier_filtered[spike_sample + 14]),
    xytext=(spike_time + 0.18, 15),
    arrowprops=dict(facecolor="#d62728", edgecolor="#d62728", width=1.5, headwidth=8),
    fontsize=9,
    fontweight="bold",
    color="#d62728",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="#d62728", alpha=0.9),
)
ax2.legend(loc="upper left", frameon=True, fontsize=9)

# Panel 3: Zoom on Start of Trial (Circular Edge Wrap-Around)
ax3 = axes[2]
zoom_edge = slice(0, 50)
t_edge = time_axis[zoom_edge]
ax3.plot(t_edge, spike_signal[zoom_edge], color="black", linewidth=1.0, alpha=0.4, label="Raw Signal (No event here)")
ax3.plot(t_edge, butter_filtered[zoom_edge], color="#1f77b4", linewidth=2.2, label="Butterworth: Undisturbed at start (0.0s)")
ax3.plot(t_edge, fourier_filtered[zoom_edge], color="#d62728", linewidth=2.0, linestyle="-", label="Fourier: Leaked Future Artifact!")
ax3.set_title("3. Zoom-in: Circular Edge Wrap-Around (Start of Trial, 0.0s to 0.4s)", fontsize=12, fontweight="bold", pad=8)
ax3.set_xlabel("Time (seconds)", fontsize=10)
ax3.set_ylabel("Amplitude (uV)", fontsize=10)
ax3.annotate(
    "Circular Leakage!\nFFT treats the signal as a loop.\nThe spike at 2.92s wrapped around\nand contaminated 0.0s!",
    xy=(0.04, fourier_filtered[5]),
    xytext=(0.08, 6),
    arrowprops=dict(facecolor="#d62728", edgecolor="#d62728", width=1.5, headwidth=8),
    fontsize=9,
    fontweight="bold",
    color="#d62728",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="#d62728", alpha=0.9),
)
ax3.legend(loc="upper right", frameon=True, fontsize=9)

plt.tight_layout()

# Save to artifacts directory and track directory
out_path_track = track_dir / "fourier_vs_butterworth.png"
out_path_brain = Path("/home/meghan/.gemini/antigravity/brain/5fc71298-bfc2-4c22-8082-f3a013e9ba88/fourier_vs_butterworth.png")

plt.savefig(out_path_track, bbox_inches="tight")
plt.savefig(out_path_brain, bbox_inches="tight")
plt.close()

print(f"\n[OK] High-resolution figure saved to:")
print(f"     1. {out_path_track}")
print(f"     2. {out_path_brain}")
print("=" * 65)

