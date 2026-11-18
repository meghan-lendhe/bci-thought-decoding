"""Study: Analyzing Subject-Specific 41x41 Covariance Networks across Test Streams.

User's Insight:
"I was thinking of studying the 41x41 cov matrix as a way of causal running calibration.
 maybe we can learn what pairs this person is using and incorporate that."

Questions to answer:
1. For each test stream (Sub 3, Sub 4, Sub 5, Sub 10):
   Which electrode pairs have the highest dynamic range / trial-to-trial variance?
   Are the active pairs the SAME across subjects, or does each person recruit different networks?
2. When performing Math vs Word, which electrode pairs actually modulate for Sub 4 vs Sub 10?
3. Can the running 41x41 covariance matrix identify a subject's personal active subnetworks causally?
"""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pyriemann.estimation import Covariances

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
n_chans = len(ch_names)
sfreq = 120.0

bands = [
    ("Theta (4-8Hz)", 4.0, 8.0),
    ("Low Alpha (8-11Hz)", 8.0, 11.0),
    ("High Alpha (11-14Hz)", 11.0, 14.0),
    ("Low Beta (14-20Hz)", 14.0, 20.0),
    ("High Beta (20-28Hz)", 20.0, 28.0),
    ("Low Gamma (28-40Hz)", 28.0, 40.0),
]

def _filter_band(X, f_low, f_high):
    sos = butter(4, [f_low, f_high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)

def _cov_to_corr(covs):
    diags = np.diagonal(covs, axis1=-2, axis2=-1)
    stds = np.sqrt(np.maximum(diags, 1e-12))[..., np.newaxis]
    corrs = covs / (stds @ stds.swapaxes(-2, -1))
    for i in range(len(corrs)):
        np.fill_diagonal(corrs[i], 1.0)
    return corrs

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

X_all_te = np.concatenate(X_all_te, axis=0)
y_all_te = np.array(y_all_te).flatten()
stream_ids = np.array(stream_ids).flatten()
unique_streams = np.unique(stream_ids)

print("=" * 90)
print("ANALYZING 41x41 COVARIANCE MODULATION ACROSS TEST STREAMS")
print("=" * 90)

cov_est = Covariances(estimator="lwf")

# For Theta (4-8Hz) and Low Beta (14-20Hz):
for bname, flow, fhigh in [("Theta (4-8Hz)", 4.0, 8.0), ("Low Beta (14-20Hz)", 14.0, 20.0), ("High Alpha (11-14Hz)", 11.0, 14.0)]:
    print(f"\n" + "#" * 90)
    print(f"FREQUENCY BAND: {bname}")
    print("#" * 90)
    
    X_filt = _filter_band(X_all_te, flow, fhigh)
    C_all = cov_est.fit_transform(X_filt[:, :, 120:360]) # 1.0 - 3.0s (active cognitive window)
    R_all = _cov_to_corr(C_all) # (755, 41, 41)
    
    for sid in unique_streams:
        mask = stream_ids == sid
        y_sub = y_all_te[mask]
        R_sub = R_all[mask] # (189, 41, 41)
        
        # 1. Variance of each pair across all 189 trials for this subject
        pair_var = np.var(R_sub, axis=0) # (41, 41)
        
        # Find top 3 most dynamically variable pairs
        var_pairs = []
        for i in range(n_chans):
            for j in range(i + 1, n_chans):
                var_pairs.append((pair_var[i, j], ch_names[i], ch_names[j]))
        var_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # 2. Difference between Math (1) and Word (2) for this subject
        m_math = y_sub == 1
        m_word = y_sub == 2
        
        diff_mw = np.mean(R_sub[m_math], axis=0) - np.mean(R_sub[m_word], axis=0)
        
        diff_pairs = []
        for i in range(n_chans):
            for j in range(i + 1, n_chans):
                diff_pairs.append((diff_mw[i, j], ch_names[i], ch_names[j]))
        
        diff_pairs.sort(key=lambda x: x[0], reverse=True)
        top_math = diff_pairs[:3]
        top_word = diff_pairs[-3:]
        
        print(f"\n--- Stream {sid} (N={mask.sum()} trials) ---")
        print("  Top 3 Most Dynamic Pairs (Highest Variance across trials):")
        for v, c1, c2 in var_pairs[:3]:
            print(f"    • {c1:<4} <-> {c2:<4} | Var: {v:.4f} (Std: {np.sqrt(v):.3f})")
            
        print("  Top 3 Couplings for Mental Subtraction (Math > Word):")
        for d, c1, c2 in top_math:
            print(f"    • {c1:<4} <-> {c2:<4} | Δr: {d:+.4f}")
            
        print("  Top 3 Couplings for Word Generation (Word > Math):")
        for d, c1, c2 in reversed(top_word):
            print(f"    • {c1:<4} <-> {c2:<4} | Δr: {-d:+.4f} (Word favored)")

