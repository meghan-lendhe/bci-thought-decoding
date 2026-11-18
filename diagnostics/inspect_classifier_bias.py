"""Inspect Ridge Classifier Intercepts and Decision Margins.

Reveals:
1. What are the baseline intercepts (b) for each class?
2. What are the average decision margins output across the test stream?
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

print("=" * 65)
print("INSPECTING CLASSIFIER BASELINE BIAS & MARGINS")
print("=" * 65)

# 1. Inspect Intercepts (b)
class_names = ["0: Motor Imagery", "1: Mental Subtraction", "2: Word Generation"]
print("\n1. RIDGE CLASSIFIER INTERCEPTS (b):")
print("-" * 65)
for i, name in enumerate(class_names):
    print(f"• {name:<24}: b = {ridge.intercept_[i]:+.4f}")

# 2. Inspect Weight Norms (How aggressive is the model on each class?)
weight_norms = np.linalg.norm(ridge.coef_, axis=1)
print("\n2. RIDGE WEIGHT VECTOR NORMS (||w||):")
print("-" * 65)
for i, name in enumerate(class_names):
    print(f"• {name:<24}: ||w|| = {weight_norms[i]:.4f}")

# 3. Stream through test set and measure average margins
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
all_margins = []
y_true_all = []

# Modify predict slightly to capture margins
split_idx = 480 // 2

# Hook into model to record ridge margins during streaming
recorded_margins = []
original_ridge_df = model.ridge.decision_function

def tracking_df(flat):
    m = original_ridge_df(flat)
    recorded_margins.append(m[0])
    return m

model.ridge.decision_function = tracking_df

for X_b, y_b, info in loaders["test"]:
    y_true_all.append(int(y_b.cpu().numpy()[0]))
    pred = model.predict(X_b)

margins_arr = np.array(recorded_margins)  # (755, 3)
y_true_arr = np.array(y_true_all)

print("3. AVERAGE DECISION MARGIN OUTPUT ACROSS ALL 755 TEST TRIALS:")
print("-" * 65)
for i, name in enumerate(class_names):
    avg_m = margins_arr[:, i].mean()
    print(f"• {name:<24}: Mean Decision Score = {avg_m:+.4f}")

print("\n4. DECISION SCORE WHEN TRUE CLASS IS WORD GENERATION (Class 2):")
print("-" * 65)
word_mask = (y_true_arr == 2)
for i, name in enumerate(class_names):
    score = margins_arr[word_mask, i].mean()
    print(f"• Model's score for {name:<22}: {score:+.4f}")

print("=" * 65)

