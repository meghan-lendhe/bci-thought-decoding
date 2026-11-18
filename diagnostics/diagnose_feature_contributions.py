"""Diagnostic Tool 2: Feature Decomposition.

Breaks down the -0.7460 decision score for Word Generation:
Which frequency band or time feature is actively killing the score?
- Theta contribution?
- Alpha contribution?
- Beta contribution?
- Gamma contribution?
- Delta Power (Late - Early) contribution?
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

# Extract weights for Class 2 (Word Generation)
# Total features = 11,070:
# 12 tokens * 902 features = 10,824 token features
# 246 delta features
w_word = ridge.coef_[2]  # (11,070,)

# Group feature indices
# 6 bands * 2 epochs = 12 tokens
band_names = [
    "1. Theta (4-8 Hz)",
    "2. Low Alpha (8-11 Hz)",
    "3. High Alpha (11-14 Hz)",
    "4. Low Beta (14-20 Hz)",
    "5. High Beta (20-28 Hz)",
    "6. Low Gamma (28-40 Hz)",
]

# Track test stream features
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
recorded_scaled_features = []
y_true_all = []

# Hook scaler.transform to capture the 11,070-dim feature vector x
orig_scaler_transform = model.scaler.transform

def tracking_scaler(flat):
    x_sc = orig_scaler_transform(flat)
    recorded_scaled_features.append(x_sc[0])
    return x_sc

model.scaler.transform = tracking_scaler

for X_b, y_b, info in loaders["test"]:
    y_true_all.append(int(y_b.cpu().numpy()[0]))
    model.predict(X_b)

X_test_scaled = np.array(recorded_scaled_features)  # (755, 11,070)
y_true_arr = np.array(y_true_all)

# Look at trials that were ACTUALLY Word Generation (Class 2)
word_trials = X_test_scaled[y_true_arr == 2]  # (254, 11,070)

# Calculate w * x for each feature component on Word Generation trials
# w * x = product of weight and feature
wx = word_trials * w_word  # (254, 11,070)
mean_wx = wx.mean(axis=0)  # average contribution across all Word trials

print("=" * 65)
print("DECOMPOSING THE WORD GENERATION SCORE BY INGREDIENT")
print("=" * 65)
print(f"Total Intercept b: {ridge.intercept_[2]:+.4f}")
print(f"Total Average Score on Word Trials: {mean_wx.sum() + ridge.intercept_[2]:+.4f}")
print("-" * 65)
print("BREAKDOWN BY FREQUENCY BAND (Tokens = Log-Power + Tangent Correlation):")
print("-" * 65)

token_dim = 902
for b_idx, b_name in enumerate(band_names):
    # Each band has 2 tokens: Early and Late
    early_start = (b_idx * 2) * token_dim
    early_end = early_start + token_dim
    late_start = (b_idx * 2 + 1) * token_dim
    late_end = late_start + token_dim
    
    contrib_early = mean_wx[early_start:early_end].sum()
    contrib_late = mean_wx[late_start:late_end].sum()
    total_band = contrib_early + contrib_late
    
    print(f"• {b_name:<26}: Total = {total_band:+.4f} (Early: {contrib_early:+.4f}, Late: {contrib_late:+.4f})")

# Breakdown of Delta Power
delta_start = 12 * token_dim  # 10,824
delta_end = delta_start + 246
contrib_delta = mean_wx[delta_start:delta_end].sum()

print("-" * 65)
print(f"• 7. Delta Power Trajectory (Late - Early): Total = {contrib_delta:+.4f}")
print("=" * 65)

