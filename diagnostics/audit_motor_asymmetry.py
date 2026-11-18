"""Diagnostic: Auditing Sensorimotor Lateralization (C3 vs C4) and Beta Rebound.

Hypothesis:
In previous models, we lumped C1, C2, C3, C4, C5, C6, Cz into a single average 'pM'.
This destroyed the contralateral motor dipole (C3 - C4 asymmetry)!

1. Does Motor Imagery show sharp C3 - C4 asymmetry compared to Math and Word?
2. Does Motor Imagery show sensorimotor Beta Rebound in Stage 4 (3.0 - 4.0s)?
3. What is the effect on the 50-70 trials currently confused between Motor and Cognitive?
"""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt

track_dir = Path("/home/meghan/projects/bci_challenge/2026-competition/tracks/bci_decoding")
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task
from benchmark_utils.data import to_numpy

data_dir = track_dir / "data" / "neural_compet"
loaders, meta = load_task(
    "eeg",
    "_motor_imagery_stream",
    data_dir=data_dir,
    device="cpu",
    batch_size=1,
    seed=42,
    num_workers=0,
    subset="full",
    target_transform=lambda y: y.argmax(-1) if hasattr(y, "argmax") else y,
)

ch_names = "C2,CP2,P2,Pz,CPz,Cz,C1,CP1,P1,P3,PO7,P5,TP7,CP5,CP3,C3,PO8,P6,P4,TP8,CP6,CP4,C6,C4,FC6,FC4,FC2,F6,F4,F2,AF4,AF3,F5,F3,F1,Fz,FC5,FC3,FC1,FCz,C5".split(",")
sfreq = 120.0

idx_c3 = ch_names.index("C3")
idx_c4 = ch_names.index("C4")
idx_cz = ch_names.index("Cz")

# Mu (11-14Hz) and Low Beta (14-20Hz)
def _filter_band(X, f_low, f_high):
    sos = butter(4, [f_low, f_high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)

# Load test data
X_all_te, y_all_te, stream_ids = [], [], []
for X, y, info in loaders["test"]:
    X_np = to_numpy(X)
    if X_np.ndim == 2:
        X_np = X_np[np.newaxis, ...]
    X_all_te.append(X_np)
    y_np = to_numpy(y)
    y_all_te.append(int(y_np.argmax(-1)) if y_np.ndim > 1 else int(y_np.flatten()[0]))
    sid = info["stream_id"][0] if isinstance(info["stream_id"], list) else info["stream_id"]
    stream_ids.append(sid)

X_all_te = np.concatenate(X_all_te, axis=0) # (755, 41, 480)
y_all_te = np.array(y_all_te).flatten()

print("=" * 90)
print("AUDITING SENSORIMOTOR ASYMMETRY (C3 vs C4) AND BETA REBOUND ACROSS CLASSES")
print("=" * 90)

for bname, flow, fhigh in [("Mu (11-14Hz)", 11.0, 14.0), ("Low Beta (14-20Hz)", 14.0, 20.0), ("High Beta (20-28Hz)", 20.0, 28.0)]:
    X_filt = _filter_band(X_all_te, flow, fhigh)
    
    # Power in 4 stages for C3, C4, Cz
    # T1: 24-120 (0.2-1.0s), T2: 120-240 (1-2s), T3: 240-360 (2-3s), T4: 360-480 (3-4s)
    p_c3 = [np.mean(X_filt[:, idx_c3, t0:t1]**2, axis=-1) for t0, t1 in [(24, 120), (120, 240), (240, 360), (360, 480)]]
    p_c4 = [np.mean(X_filt[:, idx_c4, t0:t1]**2, axis=-1) for t0, t1 in [(24, 120), (120, 240), (240, 360), (360, 480)]]
    p_cz = [np.mean(X_filt[:, idx_cz, t0:t1]**2, axis=-1) for t0, t1 in [(24, 120), (120, 240), (240, 360), (360, 480)]]
    
    # C3 vs C4 Asymmetry in active stage 2 & 3:
    # 10 * log10(C3 / C4)
    asym_s2 = 10.0 * np.log10((p_c3[1] + 1e-12) / (p_c4[1] + 1e-12))
    asym_s3 = 10.0 * np.log10((p_c3[2] + 1e-12) / (p_c4[2] + 1e-12))
    
    # Motor Asymmetry Magnitude: |C3 - C4| (Either right-hand or left-hand will be non-zero!)
    # While diffuse cognitive effort should have C3 ~ C4 => |asym| ~ 0
    asym_mag_s2 = np.abs(asym_s2)
    asym_mag_s3 = np.abs(asym_s3)
    
    # Beta Rebound: Stage 4 (3-4s) minus Stage 3 (2-3s) [dB]
    rebound_c3 = 10.0 * np.log10((p_c3[3] + 1e-12) / (p_c3[2] + 1e-12))
    rebound_c4 = 10.0 * np.log10((p_c4[3] + 1e-12) / (p_c4[2] + 1e-12))
    rebound_cz = 10.0 * np.log10((p_cz[3] + 1e-12) / (p_cz[2] + 1e-12))
    mean_motor_rebound = (rebound_c3 + rebound_c4 + rebound_cz) / 3.0
    
    print(f"\n--- Band: {bname} ---")
    for c, cname in [(0, "Motor Imagery (0)"), (1, "Mental Sub (1)"), (2, "Word Gen (2)")]:
        m = y_all_te == c
        mag_s2 = np.mean(asym_mag_s2[m])
        mag_s3 = np.mean(asym_mag_s3[m])
        reb = np.mean(mean_motor_rebound[m])
        print(f"  • {cname:<20} | |C3-C4| Asymmetry: S2={mag_s2:.3f} dB, S3={mag_s3:.3f} dB | Late Rebound (S4-S3): {reb:+.3f} dB")

