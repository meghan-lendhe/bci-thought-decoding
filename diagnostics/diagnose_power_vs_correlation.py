"""Diagnostic: Dissecting Power vs Riemannian Correlation & Trajectory Directions.

Answers:
1. In the -0.1688 Low Alpha deficit, how much is Power (41 feats) vs Tangent Correlation (861 feats)?
2. Does Riemannian Distance actually depend on power?
3. How does Delta (Late - Early) break down across all 6 bands?
4. What is the direction (+/-) of Delta for Word Generation?
"""

import sys
import importlib.util
from pathlib import Path
import numpy as np

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

output_dir = track_dir / "outputs" / "AdaptiveMasterEnsemble"
sub_file = output_dir / "submission.py"

spec = importlib.util.spec_from_file_location("sub", sub_file)
sub_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub_mod)

solver = sub_mod.Solver()
meta = {
    "submission_dir": output_dir,
    "n_chans": 41,
    "n_classes": 3,
    "sfreq": 120.0,
    "device": "cpu",
}

model = solver.load_model(meta)
ridge = model.ridge
w_word = ridge.coef_[2]  # (11,070,)

band_names = [
    "Theta (4-8 Hz)",
    "Low Alpha (8-11 Hz)",
    "High Alpha (11-14 Hz)",
    "Low Beta (14-20 Hz)",
    "High Beta (20-28 Hz)",
    "Low Gamma (28-40 Hz)",
]

loaders, task_meta = load_task(
    "eeg",
    "_motor_imagery_stream",
    data_dir=track_dir / "data" / "neural_compet",
    dataset=None,
    device="cpu",
    batch_size=1,
    seed=42,
    num_workers=0,
    subset="full",
    target_transform=lambda y: y.argmax(-1) if hasattr(y, "argmax") else y,
)

model.reset_state()
recorded_scaled = []
recorded_raw = []
y_true_all = []

orig_scaler_transform = model.scaler.transform

def tracking_scaler(flat):
    recorded_raw.append(flat[0].copy())
    x_sc = orig_scaler_transform(flat)
    recorded_scaled.append(x_sc[0].copy())
    return x_sc

model.scaler.transform = tracking_scaler

print("Running test stream to collect trial features...")
for X_b, y_b, info in loaders["test"]:
    y_true_all.append(int(y_b.cpu().numpy()[0]))
    model.predict(X_b)

X_sc = np.array(recorded_scaled)   # (755, 11,070)
X_raw = np.array(recorded_raw)     # (755, 11,070)
y_true = np.array(y_true_all)

# Word Generation trials (Class 2)
word_sc = X_sc[y_true == 2]        # (254, 11,070)
word_raw = X_raw[y_true == 2]      # (254, 11,070)
wx = word_sc * w_word
mean_wx = wx.mean(axis=0)

token_dim = 902  # 41 power + 861 correlation
n_chans = 41
n_conn = 861

print("\n" + "=" * 75)
print("1. POWER (41 CHANS) VS RIEMANNIAN CORRELATION (861 PAIRS) BREAKDOWN")
print("=" * 75)
print(f"{'Band':<24} | {'Total Contrib':<14} | {'Power (41)':<12} | {'Riemann Corr (861)':<18}")
print("-" * 75)

for b_idx, b_name in enumerate(band_names):
    # Early token
    e_start = (b_idx * 2) * token_dim
    e_p_slice = slice(e_start, e_start + n_chans)
    e_c_slice = slice(e_start + n_chans, e_start + token_dim)
    
    # Late token
    l_start = (b_idx * 2 + 1) * token_dim
    l_p_slice = slice(l_start, l_start + n_chans)
    l_c_slice = slice(l_start + n_chans, l_start + token_dim)
    
    p_contrib = mean_wx[e_p_slice].sum() + mean_wx[l_p_slice].sum()
    c_contrib = mean_wx[e_c_slice].sum() + mean_wx[l_c_slice].sum()
    total_band = p_contrib + c_contrib
    
    print(f"{b_name:<24} | {total_band:+14.4f} | {p_contrib:+12.4f} | {c_contrib:+18.4f}")

print("\n" + "=" * 75)
print("2. DELTA POWER TRAJECTORY (Late - Early) BREAKDOWN BY ALL 6 BANDS")
print("=" * 75)
print(f"{'Band':<24} | {'Raw Mean Delta':<16} | {'Direction':<12} | {'Score Contrib (w*x)':<18}")
print("-" * 75)

delta_start = 12 * token_dim  # 10,824
for b_idx, b_name in enumerate(band_names):
    d_slice = slice(delta_start + b_idx * n_chans, delta_start + (b_idx + 1) * n_chans)
    
    # Raw Delta value (Late - Early in log space)
    raw_delta_vals = word_raw[:, d_slice].mean()
    # Direction
    direction = "INCREASE (+)" if raw_delta_vals > 0 else "DROP (-)"
    # Contribution to Word Generation score
    contrib = mean_wx[d_slice].sum()
    
    print(f"{b_name:<24} | {raw_delta_vals:+16.4f} | {direction:<12} | {contrib:+18.4f}")

print("-" * 75)
total_delta_contrib = mean_wx[delta_start:delta_start + 246].sum()
print(f"{'TOTAL DELTA CONTRIBUTION':<24} | {'':<16} | {'':<12} | {total_delta_contrib:+18.4f}")
print("=" * 75)

