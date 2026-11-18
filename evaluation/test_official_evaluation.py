"""Evaluate AdaptiveRegionalEnsemble using the OFFICIAL competition evaluator (predict_streams).

The official benchmark runner in objective.py calls predict_streams(model, test_loader).
predict_streams automatically calls model.reset_state() on every new subject/session stream!
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task
from benchmark_utils.streaming import predict_streams
from solvers.adaptive_regional_ensemble import AdaptiveRegionalEnsembleModel

print("=" * 75)
print("OFFICIAL BENCHMARK EVALUATION (USING predict_streams WITH PER-STREAM RESET)")
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

# Fit on training data
print("\n[1/3] Fitting on training data...")
model.fit(loaders["train"])

# Run official predict_streams
print("\n[2/3] Running official predict_streams(model, loaders['test'])...")
y_true, y_pred, info = predict_streams(model, loaders["test"])

y_true = np.asarray(y_true).reshape(-1)
y_pred = np.asarray(y_pred).reshape(-1)
streams = info["stream_id"]

print(f"\n[3/3] Inspecting streams and per-stream accuracy...")
unique_streams = list(dict.fromkeys(streams))
print(f"Total test windows: {len(y_true)}")
print(f"Identified {len(unique_streams)} distinct streams in test set: {unique_streams}")

classes = [0, 1, 2]
class_names = ["Motor Imagery (0)", "Mental Sub (1)   ", "Word Gen (2)     "]

for s_idx, s in enumerate(unique_streams):
    mask = (streams == s)
    s_yt = y_true[mask]
    s_yp = y_pred[mask]
    s_acc = (s_yp == s_yt).mean() * 100.0
    s_bal = balanced_accuracy_score(s_yt, s_yp) * 100.0
    s_cm = confusion_matrix(s_yt, s_yp, labels=classes)
    
    print(f"\n{'='*65}")
    print(f"STREAM {s_idx+1}: stream_id = {s} ({len(s_yt)} trials)")
    print(f"{'='*65}")
    print(f"Overall Accuracy:  {s_acc:.2f}% | Balanced Accuracy: {s_bal:.2f}%")
    print(f"{'-'*65}")
    for c, cname in zip(classes, class_names):
        c_tot = (s_yt == c).sum()
        c_cor = s_cm[c, c]
        c_acc = c_cor / c_tot * 100.0 if c_tot > 0 else 0.0
        print(f"• {cname:<22}: {c_acc:5.1f}% ({c_cor}/{c_tot})")
    print(f"Confusion Matrix:\n{s_cm}")

# Overall pooled
tot_acc = (y_pred == y_true).mean() * 100.0
bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
cm = confusion_matrix(y_true, y_pred, labels=classes)

print(f"\n{'='*75}")
print("OVERALL POOLED METRICS (ACROSS ALL STREAMS WITH PROPER PER-STREAM RESET)")
print(f"{'='*75}")
print(f"Overall Accuracy:  {tot_acc:.2f}%")
print(f"Balanced Accuracy: {bal_acc:.2f}%")
print(f"Confusion Matrix (True \\ Pred):\n{cm}")
print(f"{'='*75}")

