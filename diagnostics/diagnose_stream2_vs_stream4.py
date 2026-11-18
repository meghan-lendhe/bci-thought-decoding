"""Diagnostic: Deep Comparative Analysis of Stream 2 (Subject 4, 42.3%) vs. Stream 4 (Subject 10, 65.2%).

Investigates:
1. Filter Edge Ringing Test (Testing Blind Spot Claim 2):
   - Measures whether sosfiltfilt causes artificial voltage/power inflation near t = 0.0s (0 - 250ms) across frequency bands.

2. Channel Quality & Bad Electrode Audit:
   - Audits variance, peak-to-peak amplitude, and kurtosis across all 41 channels for Subject 4 vs. Subject 10.
   - Detects if specific noisy/loose electrodes (e.g. TP7, TP8, PO7, PO8) are corrupting Subject 4.

3. Task-Specific Chronometric Activation Comparison:
   - Traces the actual 4-stage trajectories (T1 -> T2 -> T3 -> T4) for:
     • Motor trials: Motor cortex Mu rhythm (11-14 Hz)
     • Math trials: Parietal cortex power across all 6 subbands (Theta, Low Alpha, Mu, Low Beta, High Beta, Gamma)
     • Word trials: Frontal cortex lexical burst (Theta & Gamma)
   - Discovers where Subject 4's calculation signal diverges from Subject 10!
"""

import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy import stats

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task
from benchmark_utils.data import to_numpy

print("=" * 80)
print("DIAGNOSTIC: STREAM 2 (SUBJECT 4) VS. STREAM 4 (SUBJECT 10) & FILTER RINGING AUDIT")
print("=" * 80)

# Load test data
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

ch_names = "C2,CP2,P2,Pz,CPz,Cz,C1,CP1,P1,P3,PO7,P5,TP7,CP5,CP3,C3,PO8,P6,P4,TP8,CP6,CP4,C6,C4,FC6,FC4,FC2,F6,F4,F2,AF4,AF3,F5,F3,F1,Fz,FC5,FC3,FC1,FCz,C5".split(",")
sfreq = 120.0

f_chs = ['FC6', 'FC4', 'FC2', 'F6', 'F4', 'F2', 'AF4', 'AF3', 'F5', 'F3', 'F1', 'Fz', 'FC5', 'FC3', 'FC1', 'FCz']
m_chs = ['C2', 'Cz', 'C1', 'C3', 'C6', 'C4', 'C5']
p_chs = ['CP2', 'P2', 'Pz', 'CPz', 'CP1', 'P1', 'P3', 'PO7', 'P5', 'CP5', 'CP3', 'PO8', 'P6', 'P4', 'CP6', 'CP4']

f_idx = [ch_names.index(ch) for ch in f_chs]
m_idx = [ch_names.index(ch) for ch in m_chs]
p_idx = [ch_names.index(ch) for ch in p_chs]

# Collect all test trials grouped by stream_id
X_by_stream = {}
y_by_stream = {}

for X, y, info in loaders["test"]:
    sid = info["stream_id"][0]
    if hasattr(sid, "item"):
        sid = sid.item()
    if sid not in X_by_stream:
        X_by_stream[sid] = []
        y_by_stream[sid] = []
    X_by_stream[sid].append(to_numpy(X)[0])
    y_val = to_numpy(y)[0]
    if hasattr(y_val, "argmax") and y_val.ndim > 0:
        y_val = y_val.argmax()
    y_by_stream[sid].append(int(y_val))

for sid in X_by_stream:
    X_by_stream[sid] = np.array(X_by_stream[sid])  # (N_s, 41, 480)
    y_by_stream[sid] = np.array(y_by_stream[sid])

s2_id = 4   # Stream 2 (Subject 4, 42.3%)
s4_id = 10  # Stream 4 (Subject 10, 65.2%)

X_s2, y_s2 = X_by_stream[s2_id], y_by_stream[s2_id]
X_s4, y_s4 = X_by_stream[s4_id], y_by_stream[s4_id]

print(f"Loaded Stream 2 (Subject 4):  {len(X_s2)} trials | Classes: {np.bincount(y_s2)}")
print(f"Loaded Stream 4 (Subject 10): {len(X_s4)} trials | Classes: {np.bincount(y_s4)}")

# =========================================================================
# AUDIT 1: FILTER EDGE RINGING TEST (Testing Claim 2)
# =========================================================================
print("\n" + "=" * 80)
print("AUDIT 1: FILTER EDGE RINGING TEST (CLAIM 2)")
print("=" * 80)

# Check Theta (4-8 Hz), Alpha (8-14 Hz), Beta (14-28 Hz) across 100ms time bins
time_bins = [
    ("0.0 - 0.2s (Edge)", 0, 24),
    ("0.2 - 0.5s", 24, 60),
    ("0.5 - 1.0s", 60, 120),
    ("1.0 - 2.0s (Middle)", 120, 240),
    ("2.0 - 3.0s (Middle)", 240, 360),
    ("3.0 - 3.8s", 360, 456),
    ("3.8 - 4.0s (Edge)", 456, 480),
]

bands = {
    "Theta (4-8 Hz)": (4.0, 8.0),
    "Low Alpha (8-11 Hz)": (8.0, 11.0),
    "Mu (11-14 Hz)": (11.0, 14.0),
    "Beta (14-28 Hz)": (14.0, 28.0),
}

# Pool all test data for filter check
X_pool = np.concatenate([X_s2, X_s4], axis=0)

print(f"{'Band':<20} | {'0.0-0.2s (Edge)':<16} | {'0.5-1.0s':<12} | {'1.0-2.0s (Mid)':<16} | {'3.8-4.0s (Edge)':<16} | {'Edge/Mid Ratio':<14}")
print("-" * 100)

for bname, (fl, fh) in bands.items():
    sos = butter(4, [fl, fh], btype="bandpass", fs=sfreq, output="sos")
    X_f = sosfiltfilt(sos, X_pool, axis=-1)
    
    bin_vars = {}
    for tname, t0, t1 in time_bins:
        bin_vars[tname] = np.mean(X_f[..., t0:t1] ** 2)
    
    edge_start = bin_vars["0.0 - 0.2s (Edge)"]
    mid = bin_vars["1.0 - 2.0s (Middle)"]
    edge_end = bin_vars["3.8 - 4.0s (Edge)"]
    ratio = edge_start / mid
    
    print(f"{bname:<20} | {edge_start:16.4f} | {bin_vars['0.5 - 1.0s']:12.4f} | {mid:16.4f} | {edge_end:16.4f} | {ratio:13.2f}x")

# =========================================================================
# AUDIT 2: CHANNEL QUALITY & ARTIFACT AUDIT (Subject 4 vs. Subject 10)
# =========================================================================
print("\n" + "=" * 80)
print("AUDIT 2: CHANNEL NOISE & IMPEDANCE AUDIT (SUBJECT 4 VS. SUBJECT 10)")
print("=" * 80)

# Broadband 1-40 Hz
sos_bb = butter(4, [1.0, 40.0], btype="bandpass", fs=sfreq, output="sos")
X_s2_bb = sosfiltfilt(sos_bb, X_s2, axis=-1)
X_s4_bb = sosfiltfilt(sos_bb, X_s4, axis=-1)

var_s2 = np.var(X_s2_bb, axis=-1).mean(axis=0)  # (41,)
var_s4 = np.var(X_s4_bb, axis=-1).mean(axis=0)  # (41,)

p2p_s2 = (np.max(X_s2_bb, axis=-1) - np.min(X_s2_bb, axis=-1)).mean(axis=0)
p2p_s4 = (np.max(X_s4_bb, axis=-1) - np.min(X_s4_bb, axis=-1)).mean(axis=0)

ratio_var = var_s2 / (var_s4 + 1e-12)

print(f"{'Channel':<8} | {'Region':<10} | {'Sub 4 Var':<12} | {'Sub 10 Var':<12} | {'Ratio (S4/S10)':<16} | {'Sub 4 P2P (uV)':<16} | {'Status':<12}")
print("-" * 92)

# Identify extreme channels
outliers_s2 = []
for c, ch in enumerate(ch_names):
    region = "Frontal" if c in f_idx else ("Motor" if c in m_idx else "Parietal")
    status = "OK"
    if ratio_var[c] > 3.0:
        status = "HIGH NOISE"
        outliers_s2.append((ch, ratio_var[c]))
    elif ratio_var[c] < 0.33:
        status = "ATTENUATED"
    print(f"{ch:<8} | {region:<10} | {var_s2[c]:12.2f} | {var_s4[c]:12.2f} | {ratio_var[c]:15.2f}x | {p2p_s2[c]:15.2f} | {status:<12}")

if outliers_s2:
    print(f"\n[ALERT] Detected {len(outliers_s2)} high-noise channels in Subject 4 (ratio > 3x):")
    for ch, r in sorted(outliers_s2, key=lambda x: -x[1]):
        print(f"  • {ch}: {r:.2f}x higher variance than Subject 10!")

# =========================================================================
# AUDIT 3: TASK ACTIVATION CHRONOMETRY (Subject 4 vs. Subject 10)
# =========================================================================
print("\n" + "=" * 80)
print("AUDIT 3: TASK ACTIVATION TRAJECTORIES ACROSS 4 STAGES (T1 -> T2 -> T3 -> T4)")
print("=" * 80)

# Check Math Trials specifically: Parietal power across 6 bands
subbands = [
    ("Theta (4-8 Hz)", 4.0, 8.0),
    ("Low Alpha (8-11 Hz)", 8.0, 11.0),
    ("Mu (11-14 Hz)", 11.0, 14.0),
    ("Low Beta (14-20 Hz)", 14.0, 20.0),
    ("High Beta (20-28 Hz)", 20.0, 28.0),
    ("Gamma (28-40 Hz)", 28.0, 40.0),
]

stages = [("T1 (0-1s)", 0, 120), ("T2 (1-2s)", 120, 240), ("T3 (2-3s)", 240, 360), ("T4 (3-4s)", 360, 480)]

print("\n--- MENTAL SUBTRACTION (CLASS 1) PARIETAL TRAJECTORY ---")
print(f"{'Band':<20} | {'Subject':<10} | {'T1 (0-1s)':<10} | {'T2 (1-2s)':<10} | {'T3 (2-3s)':<10} | {'T4 (3-4s)':<10} | {'T2->T3 Surge':<14}")
print("-" * 88)

for bname, fl, fh in subbands:
    sos = butter(4, [fl, fh], btype="bandpass", fs=sfreq, output="sos")
    
    # Subject 10 Math trials
    X_s4_filt = sosfiltfilt(sos, X_s4[y_s4 == 1], axis=-1)
    p_s4 = []
    for _, t0, t1 in stages:
        p_s4.append(np.mean(np.log(np.mean(X_s4_filt[:, p_idx, t0:t1] ** 2, axis=-1) + 1e-12)))
    surge_s4 = p_s4[2] - p_s4[1]
    
    # Subject 4 Math trials
    X_s2_filt = sosfiltfilt(sos, X_s2[y_s2 == 1], axis=-1)
    p_s2 = []
    for _, t0, t1 in stages:
        p_s2.append(np.mean(np.log(np.mean(X_s2_filt[:, p_idx, t0:t1] ** 2, axis=-1) + 1e-12)))
    surge_s2 = p_s2[2] - p_s2[1]
    
    print(f"{bname:<20} | Subject 10 | {p_s4[0]:10.3f} | {p_s4[1]:10.3f} | {p_s4[2]:10.3f} | {p_s4[3]:10.3f} | {surge_s4:+13.3f} dB")
    print(f"{'':<20} | Subject 4  | {p_s2[0]:10.3f} | {p_s2[1]:10.3f} | {p_s2[2]:10.3f} | {p_s2[3]:10.3f} | {surge_s2:+13.3f} dB")
    print("-" * 88)

# Check Motor Imagery (Class 0): Motor Mu (11-14 Hz) Drop
sos_mu = butter(4, [11.0, 14.0], btype="bandpass", fs=sfreq, output="sos")
X_s4_mu = sosfiltfilt(sos_mu, X_s4[y_s4 == 0], axis=-1)
X_s2_mu = sosfiltfilt(sos_mu, X_s2[y_s2 == 0], axis=-1)

p_m_s4 = [np.mean(np.log(np.mean(X_s4_mu[:, m_idx, t0:t1] ** 2, axis=-1) + 1e-12)) for _, t0, t1 in stages]
p_m_s2 = [np.mean(np.log(np.mean(X_s2_mu[:, m_idx, t0:t1] ** 2, axis=-1) + 1e-12)) for _, t0, t1 in stages]

print("\n--- MOTOR IMAGERY (CLASS 0) MOTOR MU (11-14 Hz) DESYNCHRONIZATION ---")
print(f"Subject 10: T1={p_m_s4[0]:.3f} -> T2={p_m_s4[1]:.3f} (Δ = {p_m_s4[1]-p_m_s4[0]:+.3f} dB) | Net T1->T4 = {p_m_s4[3]-p_m_s4[0]:+.3f} dB")
print(f"Subject 4:  T1={p_m_s2[0]:.3f} -> T2={p_m_s2[1]:.3f} (Δ = {p_m_s2[1]-p_m_s2[0]:+.3f} dB) | Net T1->T4 = {p_m_s2[3]-p_m_s2[0]:+.3f} dB")

# Check Word Generation (Class 2): Frontal Theta/Gamma Early Burst
sos_th = butter(4, [4.0, 8.0], btype="bandpass", fs=sfreq, output="sos")
X_s4_th = sosfiltfilt(sos_th, X_s4[y_s4 == 2], axis=-1)
X_s2_th = sosfiltfilt(sos_th, X_s2[y_s2 == 2], axis=-1)

p_f_s4 = [np.mean(np.log(np.mean(X_s4_th[:, f_idx, t0:t1] ** 2, axis=-1) + 1e-12)) for _, t0, t1 in stages]
p_f_s2 = [np.mean(np.log(np.mean(X_s2_th[:, f_idx, t0:t1] ** 2, axis=-1) + 1e-12)) for _, t0, t1 in stages]

print("\n--- WORD GENERATION (CLASS 2) FRONTAL THETA (4-8 Hz) TRAJECTORY ---")
print(f"Subject 10: T1={p_f_s4[0]:.3f} -> T2={p_f_s4[1]:.3f} (Δ = {p_f_s4[1]-p_f_s4[0]:+.3f} dB) | Net T1->T4 = {p_f_s4[3]-p_f_s4[0]:+.3f} dB")
print(f"Subject 4:  T1={p_f_s2[0]:.3f} -> T2={p_f_s2[1]:.3f} (Δ = {p_f_s2[1]-p_f_s2[0]:+.3f} dB) | Net T1->T4 = {p_f_s2[3]-p_f_s2[0]:+.3f} dB")
print("=" * 80)

