"""Diagnostic: Dissecting the 160 Math vs Word Confusion Trials.

Compares 4 Groups:
1. Group C (Correct Word): True Word -> Pred Word (~102 trials)
2. Group A (Confused Word): True Word -> Pred Math (~92 trials)
3. Group D (Correct Math): True Math -> Pred Math (~137 trials)
4. Group B (Confused Math): True Math -> Pred Word (~68 trials)

Measures across all 4 groups:
- Regional Contrast: Power(Language Subsystem) - Power(Calculation Subsystem)
- Theta Power (4-8 Hz) vs Low Gamma Power (28-40 Hz)
- Theta / Gamma Ratio
- Temporal Trajectory: Power(Late) - Power(Early)
- Decision Margins: Score(Math) vs Score(Word)
"""

import sys
from pathlib import Path
import numpy as np

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task
from benchmark_utils.streaming import predict_streams
from solvers.adaptive_regional_ensemble import AdaptiveRegionalEnsembleModel

print("=" * 75)
print("DIAGNOSING THE 160 MATH VS WORD CONFUSION ERRORS")
print("=" * 75)

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

model = AdaptiveRegionalEnsembleModel(
    n_chans=meta.get("n_chans", 41),
    n_classes=meta.get("n_classes", 3),
    sfreq=meta.get("sfreq", 120.0),
    alpha=750.0,
    prior_N0=15,
    device="cpu",
)

# 1. Fit on Training Data
print("\n[1/3] Fitting Adaptive Regional Ensemble...")
model.fit(loaders["train"])

# 2. Run Test Stream using Official predict_streams
print("\n[2/3] Streaming through Test Set with predict_streams...")
y_true, y_pred, info = predict_streams(model, loaders["test"])

y_true = np.asarray(y_true).reshape(-1)
y_pred = np.asarray(y_pred).reshape(-1)

# 3. Extract Raw Window Features for All 755 Test Trials
print("\n[3/3] Extracting Neurophysiological Signal Metrics across all test trials...")
raw_windows = []
for X_b, _, _ in loaders["test"]:
    raw_windows.append(X_b.cpu().numpy()[0])  # (41, 480)

raw_windows = np.array(raw_windows)  # (755, 41, 480)

# Channel index groups
lang_idx = model.roi_indices[0]  # 10 channels
calc_idx = model.roi_indices[2]  # 14 channels

# Band indices:
# Band 0: Theta (4-8 Hz)
# Band 5: Low Gamma (28-40 Hz)
split = 240  # 480 // 2

# Compute per-trial signal metrics
theta_e = model._filter_band(raw_windows[..., :split], 4.0, 8.0)
theta_l = model._filter_band(raw_windows[..., split:], 4.0, 8.0)
gamma_e = model._filter_band(raw_windows[..., :split], 28.0, 40.0)
gamma_l = model._filter_band(raw_windows[..., split:], 28.0, 40.0)

# Log powers: mean over time axis
# Language Subsystem Theta & Gamma
p_theta_lang = np.log(np.mean(theta_e[:, lang_idx] ** 2, axis=-1).mean(axis=-1) + 1e-12)
p_gamma_lang = np.log(np.mean(gamma_e[:, lang_idx] ** 2, axis=-1).mean(axis=-1) + 1e-12)

# Calculation Subsystem Theta & Gamma
p_theta_calc = np.log(np.mean(theta_e[:, calc_idx] ** 2, axis=-1).mean(axis=-1) + 1e-12)
p_gamma_calc = np.log(np.mean(gamma_e[:, calc_idx] ** 2, axis=-1).mean(axis=-1) + 1e-12)

# Regional Contrast: Power(Language) - Power(Calculation)
# In early window (0-2s) across all 6 bands
all_e = raw_windows[..., :split]
p_lang_total = np.log(np.mean(all_e[:, lang_idx] ** 2, axis=-1).mean(axis=-1) + 1e-12)
p_calc_total = np.log(np.mean(all_e[:, calc_idx] ** 2, axis=-1).mean(axis=-1) + 1e-12)
regional_contrast = p_lang_total - p_calc_total

# Temporal Trajectory (Late - Early) for Language and Calculation
p_lang_early = np.log(np.mean(all_e[:, lang_idx] ** 2, axis=-1).mean(axis=-1) + 1e-12)
p_lang_late = np.log(np.mean(raw_windows[..., split:][:, lang_idx] ** 2, axis=-1).mean(axis=-1) + 1e-12)
delta_lang = p_lang_late - p_lang_early

p_calc_early = np.log(np.mean(all_e[:, calc_idx] ** 2, axis=-1).mean(axis=-1) + 1e-12)
p_calc_late = np.log(np.mean(raw_windows[..., split:][:, calc_idx] ** 2, axis=-1).mean(axis=-1) + 1e-12)
delta_calc = p_calc_late - p_calc_early

# Define the 4 target groups
group_correct_word = (y_true == 2) & (y_pred == 2)  # Group C
group_confused_word = (y_true == 2) & (y_pred == 1) # Group A: True Word -> Pred Math
group_correct_math = (y_true == 1) & (y_pred == 1)  # Group D
group_confused_math = (y_true == 1) & (y_pred == 2) # Group B: True Math -> Pred Word

groups = [
    ("Group C: Correct Word (True Word -> Pred Word)", group_correct_word),
    ("Group A: Confused Word (True Word -> Pred Math)", group_confused_word),
    ("Group D: Correct Math (True Math -> Pred Math)", group_correct_math),
    ("Group B: Confused Math (True Math -> Pred Word)", group_confused_math),
]

print("\n" + "=" * 80)
print("SIDE-BY-SIDE COMPARISON: WHAT MAKES A CONFUSED TRIAL FAIL?")
print("=" * 80)
print(f"{'Metric':<34} | {'Correct Word':<12} | {'Confused Word':<14} | {'Correct Math':<12} | {'Confused Math':<14}")
print("-" * 80)

def print_row(name, data):
    vals = [data[mask].mean() if mask.sum() > 0 else 0.0 for _, mask in groups]
    print(f"{name:<34} | {vals[0]:+12.4f} | {vals[1]:+14.4f} | {vals[2]:+12.4f} | {vals[3]:+14.4f}")

# Trial counts
counts = [mask.sum() for _, mask in groups]
print(f"{'Trial Count':<34} | {counts[0]:12d} | {counts[1]:14d} | {counts[2]:12d} | {counts[3]:14d}")
print("-" * 80)

# 1. Regional Contrast
print_row("1. Regional Contrast (Lang - Calc)", regional_contrast)

# 2. Theta Power
print_row("2. Frontal Theta Power (4-8 Hz)", p_theta_lang)
print_row("   Parietal Theta Power (4-8 Hz)", p_theta_calc)

# 3. Low Gamma Power
print_row("3. Frontal Gamma Power (28-40 Hz)", p_gamma_lang)
print_row("   Parietal Gamma Power (28-40 Hz)", p_gamma_calc)

# 4. Theta / Gamma Ratio (Log difference)
ratio_lang = p_theta_lang - p_gamma_lang
ratio_calc = p_theta_calc - p_gamma_calc
print_row("4. Theta - Gamma Ratio (Frontal)", ratio_lang)
print_row("   Theta - Gamma Ratio (Parietal)", ratio_calc)

# 5. Temporal Trajectories
print("-" * 80)
print_row("5. Lang Trajectory (Late - Early)", delta_lang)
print_row("   Calc Trajectory (Late - Early)", delta_calc)

print("=" * 80)

