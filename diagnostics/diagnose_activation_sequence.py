"""Diagnostic: Testing the Regional Activation Sequence Hypothesis.

Tests whether the 3 task classes (Motor Imagery, Mental Subtraction, Word Generation)
exhibit distinct temporal chronometry and directed information flow across the 4-second window.

Measures:
1. Multi-Stage Chronometry (4 sequential 1-second epochs: T1 [0-1s], T2 [1-2s], T3 [2-3s], T4 [3-4s]):
   - Frontal Power across T1, T2, T3, T4
   - Parietal Power across T1, T2, T3, T4
   - Frontal-Parietal Balance (Frontal - Parietal) across T1, T2, T3, T4
   - State-to-state slopes (T1->T2, T2->T3, T3->T4)
2. Directed Time-Lagged Functional Connectivity (tau = 80ms / 10 samples at 120Hz):
   - R(Frontal -> Parietal) vs. R(Parietal -> Frontal)
   - Net Directed Asymmetry (Positive = Frontal leads, Negative = Parietal leads)
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
print("DIAGNOSTIC: TESTING THE REGIONAL ACTIVATION SEQUENCE HYPOTHESIS")
print("=" * 80)

data_dir = track_dir / "data" / "neural_compet"
loaders, meta = load_task(
    "eeg",
    "_motor_imagery_stream",
    data_dir=data_dir,
    dataset=None,
    device="cpu",
    batch_size=32,
    seed=42,
    num_workers=0,
    subset="full",
    target_transform=lambda y: y.argmax(-1) if hasattr(y, "argmax") else y,
)

ch_names = "C2,CP2,P2,Pz,CPz,Cz,C1,CP1,P1,P3,PO7,P5,TP7,CP5,CP3,C3,PO8,P6,P4,TP8,CP6,CP4,C6,C4,FC6,FC4,FC2,F6,F4,F2,AF4,AF3,F5,F3,F1,Fz,FC5,FC3,FC1,FCz,C5".split(",")

# Define Anatomical Clusters
frontal_chs = [ch for ch in ch_names if ch.startswith("F") or ch.startswith("AF")]
parietal_chs = [ch for ch in ch_names if ch.startswith("P") or ch.startswith("PO") or ch.startswith("CP")]
motor_chs = [ch for ch in ch_names if ch.startswith("C") and not ch.startswith("CP")]

f_idx = [ch_names.index(ch) for ch in frontal_chs]
p_idx = [ch_names.index(ch) for ch in parietal_chs]
m_idx = [ch_names.index(ch) for ch in motor_chs]

print(f"Frontal Cluster ({len(f_idx)} chs):  {frontal_chs}")
print(f"Parietal Cluster ({len(p_idx)} chs): {parietal_chs}")
print(f"Motor Cluster ({len(m_idx)} chs):    {motor_chs}")

# Collect all training and test trials
print("\n[1/3] Loading and pooling all trials (1,521 train + 755 test)...")
X_list, y_list = [], []
for loader_name in ["train", "test"]:
    for X, y, _ in loaders[loader_name]:
        X_list.append(to_numpy(X))
        y_np = to_numpy(y)
        if y_np.ndim > 1:
            y_list.append(y_np.argmax(axis=-1))
        else:
            y_list.append(y_np)

X_all = np.concatenate(X_list, axis=0)  # (N, 41, 480)
y_all = np.concatenate(y_list, axis=0)  # (N,)

sfreq = 120.0
n_samples = X_all.shape[-1]  # 480
w_len = 120  # 1.0 second per stage (4 stages total)

# 4 stages
stages = [
    ("T1 (0-1s)", 0, 120),
    ("T2 (1-2s)", 120, 240),
    ("T3 (2-3s)", 240, 360),
    ("T4 (3-4s)", 360, 480),
]

print(f"\n[2/3] Analyzing 4-Stage Chronometry for {len(X_all)} trials across 3 classes...")

# Bandpass filter for broadband EEG signal excluding DC drift and high-freq noise (1 - 40 Hz)
sos = butter(4, [1.0, 40.0], btype="bandpass", fs=sfreq, output="sos")
X_filt = sosfiltfilt(sos, X_all, axis=-1)

# Compute Log-Power per stage for Frontal, Parietal, and Motor
# Shape: (N, 4)
p_front = np.zeros((len(X_all), 4))
p_pariet = np.zeros((len(X_all), 4))
p_motor = np.zeros((len(X_all), 4))

for s, (_, start, end) in enumerate(stages):
    seg = X_filt[..., start:end]
    p_front[:, s] = np.log(np.mean(seg[:, f_idx] ** 2, axis=(1, 2)) + 1e-12)
    p_pariet[:, s] = np.log(np.mean(seg[:, p_idx] ** 2, axis=(1, 2)) + 1e-12)
    p_motor[:, s] = np.log(np.mean(seg[:, m_idx] ** 2, axis=(1, 2)) + 1e-12)

# Frontal - Parietal Balance across stages
fp_balance = p_front - p_pariet  # (N, 4)

# Slopes (State transitions)
slope_1_2 = fp_balance[:, 1] - fp_balance[:, 0]  # T1 -> T2
slope_2_3 = fp_balance[:, 2] - fp_balance[:, 1]  # T2 -> T3
slope_3_4 = fp_balance[:, 3] - fp_balance[:, 2]  # T3 -> T4
slope_overall = fp_balance[:, 3] - fp_balance[:, 0]  # T1 -> T4

# [3/3] Directed Time-Lagged Cross-Correlation (tau = 80ms / 10 samples)
tau = 10  # 10 samples @ 120Hz = 83.3 ms
print(f"[3/3] Computing Directed Frontal <-> Parietal Conduction Flow (tau = {tau/sfreq*1000:.1f}ms)...")

flow_directed = np.zeros(len(X_all))
for i in range(len(X_all)):
    # Mean frontal and parietal time series
    sig_f = np.mean(X_filt[i, f_idx], axis=0)  # (480,)
    sig_p = np.mean(X_filt[i, p_idx], axis=0)  # (480,)

    # Normalize
    sig_f = (sig_f - sig_f.mean()) / (sig_f.std() + 1e-12)
    sig_p = (sig_p - sig_p.mean()) / (sig_p.std() + 1e-12)

    # R_FP(tau): Frontal(t) * Parietal(t + tau) -> Frontal leads Parietal
    r_fp = np.mean(sig_f[:-tau] * sig_p[tau:])
    # R_PF(tau): Parietal(t) * Frontal(t + tau) -> Parietal leads Frontal
    r_pf = np.mean(sig_p[:-tau] * sig_f[tau:])

    flow_directed[i] = r_fp - r_pf

# Class masks
mask_mi = y_all == 0
mask_math = y_all == 1
mask_word = y_all == 2

classes = [
    ("Motor Imagery (0)", mask_mi),
    ("Mental Subtraction (1)", mask_math),
    ("Word Generation (2)", mask_word),
]

print("\n" + "=" * 80)
print("RESULTS: TEMPORAL CHRONOMETRY ACROSS THE 4 STAGES")
print("=" * 80)
print(f"{'Metric':<36} | {'Motor Imagery':<13} | {'Mental Sub':<13} | {'Word Gen':<13}")
print("-" * 80)

def row(name, data):
    v = [data[m].mean() for _, m in classes]
    print(f"{name:<36} | {v[0]:+13.4f} | {v[1]:+13.4f} | {v[2]:+13.4f}")

# Frontal - Parietal Balance at each stage
row("Balance T1 [0-1s] (Front - Pariet)", fp_balance[:, 0])
row("Balance T2 [1-2s] (Front - Pariet)", fp_balance[:, 1])
row("Balance T3 [2-3s] (Front - Pariet)", fp_balance[:, 2])
row("Balance T4 [3-4s] (Front - Pariet)", fp_balance[:, 3])

print("-" * 80)
print("STAGE-TO-STAGE TRANSITIONS (TEMPORAL SEQUENCING)")
print("-" * 80)
row("Step 1: T1 -> T2 Transition", slope_1_2)
row("Step 2: T2 -> T3 Transition", slope_2_3)
row("Step 3: T3 -> T4 Transition", slope_3_4)
row("Full Trial: T1 -> T4 Net Drift", slope_overall)

print("-" * 80)
print("DIRECTED AXONAL FLOW (tau = 83ms: + is Frontal->Parietal, - is Parietal->Frontal)")
print("-" * 80)
row("Net Conduction Flow [F->P - P->F]", flow_directed)

# Statistical Significance Tests between Math and Word
t_t12, p_t12 = stats.ttest_ind(slope_1_2[mask_math], slope_1_2[mask_word])
t_t23, p_t23 = stats.ttest_ind(slope_2_3[mask_math], slope_2_3[mask_word])
t_t34, p_t34 = stats.ttest_ind(slope_3_4[mask_math], slope_3_4[mask_word])
t_flow, p_flow = stats.ttest_ind(flow_directed[mask_math], flow_directed[mask_word])

print("\n" + "=" * 80)
print("STATISTICAL SEPARATION (MENTAL SUBTRACTION VS. WORD GENERATION)")
print("=" * 80)
print(f"• T1 -> T2 Transition Difference: t = {t_t12:+.3f}, p = {p_t12:.4e} {'[SIGNIFICANT]' if p_t12 < 0.05 else ''}")
print(f"• T2 -> T3 Transition Difference: t = {t_t23:+.3f}, p = {p_t23:.4e} {'[SIGNIFICANT]' if p_t23 < 0.05 else ''}")
print(f"• T3 -> T4 Transition Difference: t = {t_t34:+.3f}, p = {p_t34:.4e} {'[SIGNIFICANT]' if p_t34 < 0.05 else ''}")
print(f"• Directed Conduction Flow Diff:  t = {t_flow:+.3f}, p = {p_flow:.4e} {'[SIGNIFICANT]' if p_flow < 0.05 else ''}")
print("=" * 80)

