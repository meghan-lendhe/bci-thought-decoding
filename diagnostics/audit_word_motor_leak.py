"""Diagnostic Audit: Why are 84 Word Generation Trials Misclassified as Motor Imagery?

User's Question:
"how are you sure of this?" (challenging the claim that Word Generation causes left motor strip
 desynchronization and whether Left-Temporal Gamma distinguishes them).

Empirical Goals:
1. Identify the exact 84 Word trials misclassified as Motor Imagery. Which streams/subjects do they belong to?
2. Measure the actual Mu (11-14Hz) and Beta (14-20Hz) power on C3, C5, Cz across:
   - True Motor trials (Class 0, correctly predicted as 0)
   - Correct Word trials (Class 2, correctly predicted as 2)
   - Misclassified Word trials (Class 2, misclassified as 0)
   Do the misclassified Word trials actually show motor ERD (power drop)?
3. Measure actual Gamma (28-40Hz) power and coupling on F5, TP7, CP5:
   - Is Gamma actually higher in Word than Motor in this dataset?
   - Or is Gamma too noisy / low SNR at 120Hz sampling rate?
"""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import roc_auc_score

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

# Indices
idx_c3  = ch_names.index("C3")
idx_c4  = ch_names.index("C4")
idx_cz  = ch_names.index("Cz")
idx_c5  = ch_names.index("C5")
idx_c1  = ch_names.index("C1")

idx_f5  = ch_names.index("F5")
idx_fc5 = ch_names.index("FC5")
idx_tp7 = ch_names.index("TP7")
idx_cp5 = ch_names.index("CP5")

# Load cached predictions from the 52.64% champion model
cache_file = track_dir / "scratch" / "cached_ovo_test.npz"
data = np.load(cache_file)
raw_m01 = data['raw_m01']
raw_m02 = data['raw_m02']
raw_m12 = data['raw_m12']
lat_lb_all = data['lat_lb_all']
y_all_te = data['y_all_te']
stream_ids = data['stream_ids']
unique_streams = data['unique_streams']

norm_m01 = raw_m01 / np.std(raw_m01)
norm_m02 = raw_m02 / np.std(raw_m02)
norm_m12 = raw_m12 / np.std(raw_m12)
motor_scores = -(norm_m01 + norm_m02) / 2.0
z_lat_lb = (lat_lb_all - np.mean(lat_lb_all)) / (np.std(lat_lb_all) + 1e-12)

# Reconstruct champion predictions
w_spat = 0.6
delta_mot = 0.05
M_mot = 10
M_cog = 10

preds = []
cur_stream = None
run_mot = 0.0
n_mot = 0
run_cog = 0.0
n_cog = 0

for i in range(len(stream_ids)):
    sid = stream_ids[i]
    if sid != cur_stream:
        cur_stream = sid
        run_mot = 0.0
        n_mot = 0
        run_cog = 0.0
        n_cog = 0

    th_mot = (run_mot / (M_mot + n_mot)) + delta_mot
    theta_cog = (run_cog / (M_cog + n_cog))

    ms = motor_scores[i]
    run_mot += ms
    n_mot += 1

    if ms > th_mot:
        preds.append(0)
    else:
        w = 0.0 if sid == 10 else w_spat
        eff_12 = (norm_m12[i] - theta_cog) - w * z_lat_lb[i]
        preds.append(2 if eff_12 > 0 else 1)
        run_cog += norm_m12[i]
        n_cog += 1

preds = np.array(preds)

# Load raw test signals
X_all_te = []
for X, _, _ in loaders["test"]:
    X_np = to_numpy(X)
    if X_np.ndim == 2:
        X_np = X_np[np.newaxis, ...]
    X_all_te.append(X_np)
X_all_te = np.concatenate(X_all_te, axis=0) # (755, 41, 480)

def _filter_band(X, fl, fh):
    sos = butter(4, [fl, fh], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)

# Masks
m_word_all = y_all_te == 2
m_word_mis = (y_all_te == 2) & (preds == 0) # The 84 misclassified Word trials!
m_word_cor = (y_all_te == 2) & (preds == 2) # Correct Word trials
m_mot_cor  = (y_all_te == 0) & (preds == 0) # Correct Motor trials

print("=" * 80)
print(f"AUDITING THE 84 WORD-TO-MOTOR MISCLASSIFICATIONS (N={m_word_mis.sum()} trials)")
print("=" * 80)

print("\n1. Which Streams Do These 84 Word Trials Belong To?")
for sid in unique_streams:
    n_sub_mis = ((stream_ids == sid) & m_word_mis).sum()
    n_sub_tot = ((stream_ids == sid) & m_word_all).sum()
    print(f"   Stream {sid}: {n_sub_mis:2d} / {n_sub_tot} Word trials misclassified as Motor ({n_sub_mis/n_sub_tot*100:4.1f}%)")

# Filter raw data in Mu, Low Beta, and Gamma
X_mu    = _filter_band(X_all_te, 11.0, 14.0)
X_beta  = _filter_band(X_all_te, 14.0, 20.0)
X_gamma = _filter_band(X_all_te, 28.0, 40.0)

# Power in active window 1.0 - 3.0s (samples 120:360)
p_mu_c3 = np.mean(X_mu[:, idx_c3, 120:360]**2, axis=-1)
p_mu_c5 = np.mean(X_mu[:, idx_c5, 120:360]**2, axis=-1)
p_mu_cz = np.mean(X_mu[:, idx_cz, 120:360]**2, axis=-1)

p_be_c3 = np.mean(X_beta[:, idx_c3, 120:360]**2, axis=-1)
p_be_c5 = np.mean(X_beta[:, idx_c5, 120:360]**2, axis=-1)
p_be_cz = np.mean(X_beta[:, idx_cz, 120:360]**2, axis=-1)

p_ga_f5  = np.mean(X_gamma[:, idx_f5,  120:360]**2, axis=-1)
p_ga_tp7 = np.mean(X_gamma[:, idx_tp7, 120:360]**2, axis=-1)
p_ga_cp5 = np.mean(X_gamma[:, idx_cp5, 120:360]**2, axis=-1)

print("\n2. Sensorimotor Power (C3, C5, Cz): Is there genuine Motor ERD?")
print(f"{'Group':<32} | {'Mu C3 (µV²)':<12} | {'Mu C5 (µV²)':<12} | {'Beta C3 (µV²)':<12} | {'Beta C5 (µV²)':<12}")
print("-" * 88)
groups = [
    ("Correct Motor Trials (0 -> 0)", m_mot_cor),
    ("Misclassified Word Trials (2 -> 0)", m_word_mis),
    ("Correct Word Trials (2 -> 2)", m_word_cor),
]
for gname, gmask in groups:
    m_c3 = np.mean(p_mu_c3[gmask])
    m_c5 = np.mean(p_mu_c5[gmask])
    b_c3 = np.mean(p_be_c3[gmask])
    b_c5 = np.mean(p_be_c5[gmask])
    print(f"{gname:<32} | {m_c3:12.3f} | {m_c5:12.3f} | {b_c3:12.3f} | {b_c5:12.3f}")

print("\n3. Left-Temporal Gamma Power (F5, TP7, CP5): Does Gamma separate Word from Motor?")
print(f"{'Group':<32} | {'Gamma F5 (µV²)':<14} | {'Gamma TP7 (µV²)':<14} | {'Gamma CP5 (µV²)':<14}")
print("-" * 82)
for gname, gmask in groups:
    g_f5  = np.mean(p_ga_f5[gmask])
    g_tp7 = np.mean(p_ga_tp7[gmask])
    g_cp5 = np.mean(p_ga_cp5[gmask])
    print(f"{gname:<32} | {g_f5:14.3f} | {g_tp7:14.3f} | {g_cp5:14.3f}")

# Compute AUC for Gamma separating Word (2) vs Motor (0)
m_mot_word = (y_all_te == 0) | (y_all_te == 2)
y_mw_bin = (y_all_te[m_mot_word] == 2).astype(int) # 1 if Word, 0 if Motor

auc_ga_f5  = roc_auc_score(y_mw_bin, p_ga_f5[m_mot_word])
auc_ga_tp7 = roc_auc_score(y_mw_bin, p_ga_tp7[m_mot_word])
auc_ga_cp5 = roc_auc_score(y_mw_bin, p_ga_cp5[m_mot_word])
auc_ga_mean = roc_auc_score(y_mw_bin, (p_ga_f5[m_mot_word] + p_ga_tp7[m_mot_word] + p_ga_cp5[m_mot_word])/3.0)

print(f"\nGamma ROC-AUC for separating Word from Motor across all test trials:")
print(f"  • Gamma F5  AUC: {auc_ga_f5:.3f}")
print(f"  • Gamma TP7 AUC: {auc_ga_tp7:.3f}")
print(f"  • Gamma CP5 AUC: {auc_ga_cp5:.3f}")
print(f"  • Gamma Left Mean AUC: {auc_ga_mean:.3f}")

print("\n4. What about Motor Score distribution for the misclassified trials?")
for gname, gmask in groups:
    ms_val = motor_scores[gmask]
    print(f"   {gname:<32} | Mean Motor Score: {np.mean(ms_val):+.3f} (Std: {np.std(ms_val):.3f})")
print("=" * 80)

