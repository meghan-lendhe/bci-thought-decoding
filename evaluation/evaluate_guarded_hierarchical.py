"""End-to-End Official Benchmark Evaluation for AdaptiveGuardedHierarchicalModel.

Uses official competition predict_streams harness with per-stream reset_state().
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task
from benchmark_utils.streaming import predict_streams
from solvers.adaptive_guarded_hierarchical import AdaptiveGuardedHierarchicalModel

print("=" * 80)
print("OFFICIAL BENCHMARK EVALUATION: ADAPTIVE GUARDED HIERARCHICAL MODEL")
print("=" * 80)

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

model = AdaptiveGuardedHierarchicalModel(
    n_chans=meta.get("n_chans", 41),
    n_classes=meta.get("n_classes", 3),
    sfreq=meta.get("sfreq", 120.0),
    prior_N0=25,
    base_th_raw=0.01,
    guard_raw=0.60,
    w_lat_raw=2.0,
    M_mot=10,
    M_cog=10,
    M_lat=15,
    device="cpu",
)

# 1. Fit on official train loader
print("\n[1/3] Fitting model on training data (1,521 trials)...")
model.fit(loaders["train"])

# 2. Run official streaming evaluation
print("\n[2/3] Running official predict_streams on test loader (755 trials)...")
y_true, y_pred, info = predict_streams(model, loaders["test"])

y_true = np.asarray(y_true).reshape(-1)
y_pred = np.asarray(y_pred).reshape(-1)
streams = np.asarray(info["stream_id"]).reshape(-1)

# 3. Compute Metrics
print("\n[3/3] Official Evaluation Results:")
unique_streams = list(dict.fromkeys(streams))
print(f"Total Test Trials: {len(y_true)}")
print(f"Unique Test Streams: {unique_streams}")

print("\nPer-Stream Balanced Accuracy:")
for sid in unique_streams:
    mask = streams == sid
    yt_s = y_true[mask]
    yp_s = y_pred[mask]
    acc_s = balanced_accuracy_score(yt_s, yp_s) * 100.0
    print(f"  • Stream {sid:2d}: {acc_s:.2f}% ({len(yt_s)} trials)")

pooled_balacc = balanced_accuracy_score(y_true, y_pred) * 100.0
print(f"\nPOOLED BALANCED ACCURACY: {pooled_balacc:.2f}%")

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix (True rows, Pred cols):")
print(cm)
print(f"  • Motor Recall (0): {cm[0,0] / cm[0].sum() * 100:.2f}% ({cm[0,0]}/{cm[0].sum()})")
print(f"  • Math Recall  (1): {cm[1,1] / cm[1].sum() * 100:.2f}% ({cm[1,1]}/{cm[1].sum()})")
print(f"  • Word Recall  (2): {cm[2,2] / cm[2].sum() * 100:.2f}% ({cm[2,2]}/{cm[2].sum()})")
print("=" * 80)

