"""Module 4: Does CAR add value in practice?

Compares 5-fold Cross-Validation balanced accuracy:
1. Baseline: Original Earlobe Reference (no CAR)
2. Experiment: Common Average Reference (CAR applied before covariance)

Uses Alpha-band Riemannian Tangent Space + Ridge Classifier.
"""

import sys
from pathlib import Path
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("MODULE 4 EXPERIMENT: DOES CAR ADD VALUE IN DECODING?")
print("=" * 65)

# 1. Load 128 trials
data_dir = track_dir / "data" / "neural_compet"
loaders, meta = load_task(
    "eeg",
    "_motor_imagery_stream",
    data_dir=data_dir,
    dataset=None,
    device="cpu",
    batch_size=128,
    seed=42,
    num_workers=0,
    subset="full",
    target_transform=lambda y: y.argmax(-1) if hasattr(y, "argmax") else y,
)

for X_batch, y_batch, info in loaders["train"]:
    break

X = X_batch.cpu().numpy()  # (128, 41, 480)
y = y_batch.cpu().numpy().astype(int)

# Filter in Alpha band (8-13 Hz)
sos = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")

def evaluate_pipeline(use_car=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        
        if use_car:
            # Subtract spatial mean across all 41 electrodes
            X_tr = X_tr - np.mean(X_tr, axis=1, keepdims=True)
            X_te = X_te - np.mean(X_te, axis=1, keepdims=True)
            
        X_tr_filt = sosfiltfilt(sos, X_tr, axis=-1)
        X_te_filt = sosfiltfilt(sos, X_te, axis=-1)
        
        cov_est = Covariances(estimator="lwf")
        cov_tr = cov_est.fit_transform(X_tr_filt)
        cov_te = cov_est.transform(X_te_filt)
        
        ts = TangentSpace(metric="riemann")
        feat_tr = ts.fit_transform(cov_tr)
        feat_te = ts.transform(cov_te)
        
        scaler = StandardScaler()
        feat_tr = scaler.fit_transform(feat_tr)
        feat_te = scaler.transform(feat_te)
        
        clf = RidgeClassifier(alpha=100.0)
        clf.fit(feat_tr, y_tr)
        preds = clf.predict(feat_te)
        scores.append(balanced_accuracy_score(y_te, preds))
        
    return np.mean(scores), np.std(scores)

acc_raw, std_raw = evaluate_pipeline(use_car=False)
acc_car, std_car = evaluate_pipeline(use_car=True)

print(f"\n1. Original Earlobe Reference (No CAR):")
print(f"   5-Fold CV Balanced Accuracy: {acc_raw * 100:.2f}% (+/- {std_raw * 100:.2f}%)")

print(f"\n2. Common Average Reference (CAR):")
print(f"   5-Fold CV Balanced Accuracy: {acc_car * 100:.2f}% (+/- {std_car * 100:.2f}%)")

diff = (acc_car - acc_raw) * 100
print(f"\nNet Difference: {diff:+.2f}%")
print("=" * 65)

