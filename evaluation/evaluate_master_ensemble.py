"""Evaluate the Champion Adaptive Master Ensemble (Codabench 0.620 model) on Test Streams.

Runs the full 41-channel AdaptiveMasterEnsembleModel using official predict_streams.
Displays:
1. Overall & Balanced Accuracy
2. Full Test Confusion Matrix
3. Per-Stream Performance across the 4 distinct test sessions
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task
from benchmark_utils.streaming import predict_streams
from solvers.adaptive_master_ensemble import AdaptiveMasterEnsembleModel

print("=" * 75)
print("EVALUATING CHAMPION ADAPTIVE MASTER ENSEMBLE (ALL 41 CHANNELS)")
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

model = AdaptiveMasterEnsembleModel(
    n_chans=meta.get("n_chans", 41),
    n_classes=meta.get("n_classes", 3),
    sfreq=meta.get("sfreq", 120.0),
    alpha=150.0,
    device="cpu",
)

# 1. Fit on training data
print("\n[1/3] Fitting Adaptive Master Ensemble on 1,521 training trials...")
model.fit(loaders["train"])

# 2. Run official predict_streams
print("\n[2/3] Streaming through Test Set with predict_streams...")
y_true, y_pred, info = predict_streams(model, loaders["test"])

y_true = np.asarray(y_true).reshape(-1)
y_pred = np.asarray(y_pred).reshape(-1)

classes = [0, 1, 2]
class_names = ["Motor Imagery (0)", "Mental Sub (1)", "Word Gen (2)"]
cm = confusion_matrix(y_true, y_pred, labels=classes)
acc = (y_true == y_pred).mean() * 100.0
bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0

print("\n" + "=" * 75)
print("POOLED RESULTS: ADAPTIVE MASTER ENSEMBLE (ALL 41 CHANNELS)")
print("=" * 75)
print(f"Overall Accuracy:  {acc:.2f}% ({np.sum(y_true == y_pred)}/{len(y_true)} trials)")
print(f"Balanced Accuracy: {bal_acc:.2f}%\n")

print(f"{'Class':<22} | {'Correct':<10} | {'Total':<10} | {'Accuracy':<10}")
print("-" * 58)
for c, cname in zip(classes, class_names):
    c_cor = cm[c, c]
    c_tot = cm[c].sum()
    c_acc = c_cor / c_tot * 100.0 if c_tot > 0 else 0.0
    print(f"{cname:<22} | {c_cor:<10} | {c_tot:<10} | {c_acc:6.1f}%")

print("\nFull Confusion Matrix (True \\ Pred):")
print(f"{'':<18} | {'Pred Motor':<12} | {'Pred Math':<12} | {'Pred Word':<12}")
print("-" * 58)
for c, cname in zip(classes, class_names):
    print(f"{cname:<18} | {cm[c, 0]:12d} | {cm[c, 1]:12d} | {cm[c, 2]:12d}")

# 3. Stream Breakdown
print("\n" + "=" * 75)
print("PER-STREAM BREAKDOWN (4 PARTICIPANT SESSIONS)")
print("=" * 75)

stream_ids = []
for _, _, meta_b in loaders["test"]:
    sid = meta_b.get("stream_id", [0])[0]
    if hasattr(sid, "item"):
        sid = sid.item()
    stream_ids.append(sid)

stream_ids = np.array(stream_ids)
unique_streams = np.unique(stream_ids)

for i, sid in enumerate(unique_streams):
    mask = stream_ids == sid
    yt_s = y_true[mask]
    yp_s = y_pred[mask]
    s_acc = (yt_s == yp_s).mean() * 100.0
    s_bal = balanced_accuracy_score(yt_s, yp_s) * 100.0
    s_cm = confusion_matrix(yt_s, yp_s, labels=classes)

    print(f"\n--- Stream {i+1} (id={sid}, {mask.sum()} trials) ---")
    print(f"Accuracy: {s_acc:.2f}% | Balanced Accuracy: {s_bal:.2f}%")
    for c, cname in zip(classes, class_names):
        c_cor = s_cm[c, c]
        c_tot = s_cm[c].sum()
        c_pct = c_cor / c_tot * 100.0 if c_tot > 0 else 0.0
        print(f"  • {cname:<18}: {c_pct:5.1f}% ({c_cor}/{c_tot})")
    print("Confusion Matrix:")
    print(s_cm)

print("\n" + "=" * 75)

