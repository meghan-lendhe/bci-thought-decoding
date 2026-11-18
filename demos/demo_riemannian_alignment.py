"""Module 6: The Secret Behind the 0.825 Score — Online Riemannian Alignment (RA).

Demonstrates:
1. The Cross-Subject Shift: Why a static model trained on Subject A suffers when
   evaluating Subject B.
2. The Online Alignment Fix: Tracking the running covariance mean during streaming
   inference and centering each window to the Identity matrix I.
"""

import sys
from pathlib import Path
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm, sqrtm
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("MODULE 6: THE 0.825 SECRET — ONLINE RIEMANNIAN ALIGNMENT (RA)")
print("=" * 65)

# 1. Load train and test streams
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

# Load training data
X_train_list, y_train_list = [], []
for X_b, y_b, info in loaders["train"]:
    X_train_list.append(X_b.cpu().numpy()[0])
    y_train_list.append(int(y_b.cpu().numpy()[0]))

X_train = np.array(X_train_list)  # (N_train, 41, 480)
y_train = np.array(y_train_list)

# Load test data
X_test_list, y_test_list = [], []
for X_b, y_b, info in loaders["test"]:
    X_test_list.append(X_b.cpu().numpy()[0])
    y_test_list.append(int(y_b.cpu().numpy()[0]))

X_test = np.array(X_test_list)  # (N_test, 41, 480)
y_test = np.array(y_test_list)

print(f"Loaded {len(X_train)} Training Trials and {len(X_test)} Test Stream Trials.")

# Filter in Alpha band (8-13 Hz)
sos = butter(4, [8.0, 13.0], btype="bandpass", fs=120.0, output="sos")
X_tr_filt = sosfiltfilt(sos, X_train, axis=-1)
X_te_filt = sosfiltfilt(sos, X_test, axis=-1)

# Compute raw covariances
cov_est = Covariances(estimator="lwf")
C_train = cov_est.fit_transform(X_tr_filt)
C_test = cov_est.transform(X_te_filt)

# -------------------------------------------------------------
# EXPERIMENT 1: Standard Static Model (No Test-Time Adaptation)
# -------------------------------------------------------------
ts_static = TangentSpace(metric="riemann")
feats_tr_static = ts_static.fit_transform(C_train)
feats_te_static = ts_static.transform(C_test)

scaler_static = StandardScaler()
feats_tr_static = scaler_static.fit_transform(feats_tr_static)
feats_te_static = scaler_static.transform(feats_te_static)

clf_static = RidgeClassifier(alpha=100.0)
clf_static.fit(feats_tr_static, y_train)
preds_static = clf_static.predict(feats_te_static)
acc_static = balanced_accuracy_score(y_test, preds_static)

# -------------------------------------------------------------
# EXPERIMENT 2: Riemannian Alignment (RA)
# -------------------------------------------------------------
# Step A: Align Training Subjects to Identity Matrix I
# Compute the geometric center of training data C_ref_train
C_ref_train = np.mean(C_train, axis=0)
R_tr_inv = invsqrtm(C_ref_train)

# Whiten all training covariance matrices: C_tilde = R^{-1/2} C R^{-1/2}
C_train_aligned = np.array([R_tr_inv @ c @ R_tr_inv for c in C_train])

# Train on aligned data
ts_ra = TangentSpace(metric="riemann")
feats_tr_ra = ts_ra.fit_transform(C_train_aligned)

scaler_ra = StandardScaler()
feats_tr_ra = scaler_ra.fit_transform(feats_tr_ra)

clf_ra = RidgeClassifier(alpha=100.0)
clf_ra.fit(feats_tr_ra, y_train)

# Step B: Streaming Online Alignment for the Test Stream
# As test trials stream in one-by-one, update running reference matrix C_ref_test!
running_sum = np.zeros_like(C_ref_train)
preds_ra = []

for t, c_curr in enumerate(C_test):
    # Online running mean of test subject's baseline
    running_sum += c_curr
    C_ref_test = running_sum / (t + 1)
    
    # Pre-whitening: slide the test trial to Identity matrix I!
    R_te_inv = invsqrtm(C_ref_test)
    c_aligned = R_te_inv @ c_curr @ R_te_inv
    
    # Project onto Tangent Space and predict
    feat_te = ts_ra.transform(c_aligned[np.newaxis, ...])
    feat_te = scaler_ra.transform(feat_te)
    preds_ra.append(clf_ra.predict(feat_te)[0])

preds_ra = np.array(preds_ra)
acc_ra = balanced_accuracy_score(y_test, preds_ra)

print("\n" + "=" * 65)
print("EXPERIMENTAL RESULTS ON UNSEEN TEST STREAM:")
print("=" * 65)
print(f"1. Static Baseline Model (No Adaptation):  {acc_static * 100:.2f}%")
print(f"2. Online Riemannian Alignment (Adaptive): {acc_ra * 100:.2f}%")
print(f"Net Gain from Online Adaptation:            {(acc_ra - acc_static) * 100:+.2f}%")
print("=" * 65)

