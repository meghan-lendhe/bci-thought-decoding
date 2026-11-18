"""Test-Time Adaptive Master Ensemble Demonstration.

Compares:
1. Static Champion Ensemble (MasterEnsemble without adaptation)
2. Adaptive Champion Ensemble (with reset_state and Online Riemannian Alignment)
on the official streamed Track 02 evaluation task!
"""

import sys
from pathlib import Path
import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

print("=" * 65)
print("TESTING ADAPTIVE MASTER ENSEMBLE WITH ONLINE RIEMANNIAN ALIGNMENT")
print("=" * 65)

# 1. Load data
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

X_train_list, y_train_list = [], []
for X_b, y_b, info in loaders["train"]:
    X_train_list.append(X_b.cpu().numpy()[0])
    y_train_list.append(int(y_b.cpu().numpy()[0]))

X_train = np.array(X_train_list)  # (128, 41, 480)
y_train = np.array(y_train_list)

X_test_list, y_test_list = [], []
for X_b, y_b, info in loaders["test"]:
    X_test_list.append(X_b.cpu().numpy()[0])
    y_test_list.append(int(y_b.cpu().numpy()[0]))

X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

print(f"[Data] Train trials: {len(X_train)}, Test stream trials: {len(X_test)}")

# 6 Neurophysiological Bands
bands = [
    (4.0, 8.0),    # Theta (Mental Subtraction)
    (8.0, 11.0),   # Low Alpha (Sensory gating)
    (11.0, 14.0),  # High Alpha / Mu (Motor Imagery)
    (14.0, 20.0),  # Low Beta (Motor planning)
    (20.0, 28.0),  # High Beta (Active execution)
    (28.0, 40.0),  # Low Gamma (Word Generation)
]
n_bands = len(bands)
sfreq = 120.0

def _filter_band(X, f_low, f_high):
    sos = butter(4, [f_low, f_high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)

def _cov_to_corr(covs):
    diags = np.diagonal(covs, axis1=-2, axis2=-1)
    stds = np.sqrt(np.maximum(diags, 1e-12))[..., np.newaxis]
    corrs = covs / (stds @ stds.swapaxes(-2, -1))
    for i in range(len(corrs)):
        np.fill_diagonal(corrs[i], 1.0)
    return corrs, diags

def _invsqrtm(C, eps=1e-6):
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

# Slice time
split_idx = X_train.shape[-1] // 2
X_tr_early, X_tr_late = X_train[..., :split_idx], X_train[..., split_idx:]
X_te_early, X_te_late = X_test[..., :split_idx], X_test[..., split_idx:]

# -------------------------------------------------------------
# PIPELINE 1: STATIC CHAMPION (NO ADAPTATION)
# -------------------------------------------------------------
print("\n[Running Pipeline 1: Static Champion Backbone...]")
cov_e_est = [Covariances(estimator="lwf") for _ in bands]
cov_l_est = [Covariances(estimator="lwf") for _ in bands]
ts_e_est = [TangentSpace(metric="riemann") for _ in bands]
ts_l_est = [TangentSpace(metric="riemann") for _ in bands]

tokens_tr_static = []
delta_tr_static = []

for i, (f_low, f_high) in enumerate(bands):
    X_e_b = _filter_band(X_tr_early, f_low, f_high)
    X_l_b = _filter_band(X_tr_late, f_low, f_high)
    
    cov_e = cov_e_est[i].fit_transform(X_e_b)
    corr_e, diag_e = _cov_to_corr(cov_e)
    conn_e = ts_e_est[i].fit_transform(corr_e)
    
    cov_l = cov_l_est[i].fit_transform(X_l_b)
    corr_l, diag_l = _cov_to_corr(cov_l)
    conn_l = ts_l_est[i].fit_transform(corr_l)
    
    p_e = np.log(np.maximum(diag_e, 1e-12))
    p_l = np.log(np.maximum(diag_l, 1e-12))
    p_delta = p_l - p_e
    
    tokens_tr_static.append(np.concatenate([p_e, conn_e], axis=1))
    tokens_tr_static.append(np.concatenate([p_l, conn_l], axis=1))
    delta_tr_static.append(p_delta)

flat_tr_static = np.concatenate(tokens_tr_static + delta_tr_static, axis=1)
scaler_static = StandardScaler()
flat_tr_static = scaler_static.fit_transform(flat_tr_static)

clf_static = RidgeClassifier(alpha=150.0)
clf_static.fit(flat_tr_static, y_train)

# Evaluate Static on Test Stream
tokens_te_static = []
delta_te_static = []

for i, (f_low, f_high) in enumerate(bands):
    X_e_b = _filter_band(X_te_early, f_low, f_high)
    X_l_b = _filter_band(X_te_late, f_low, f_high)
    
    cov_e = cov_e_est[i].transform(X_e_b)
    corr_e, diag_e = _cov_to_corr(cov_e)
    conn_e = ts_e_est[i].transform(corr_e)
    
    cov_l = cov_l_est[i].transform(X_l_b)
    corr_l, diag_l = _cov_to_corr(cov_l)
    conn_l = ts_l_est[i].transform(corr_l)
    
    p_e = np.log(np.maximum(diag_e, 1e-12))
    p_l = np.log(np.maximum(diag_l, 1e-12))
    p_delta = p_l - p_e
    
    tokens_te_static.append(np.concatenate([p_e, conn_e], axis=1))
    tokens_te_static.append(np.concatenate([p_l, conn_l], axis=1))
    delta_te_static.append(p_delta)

flat_te_static = scaler_static.transform(np.concatenate(tokens_te_static + delta_te_static, axis=1))
preds_static = clf_static.predict(flat_te_static)
acc_static = balanced_accuracy_score(y_test, preds_static)

# -------------------------------------------------------------
# PIPELINE 2: ADAPTIVE CHAMPION (ONLINE RIEMANNIAN ALIGNMENT)
# -------------------------------------------------------------
print("[Running Pipeline 2: Adaptive Champion with Online Riemannian Alignment...]")

# Step A: Align Training Subjects to I
# For each band and epoch, compute training baseline covariance and pre-whiten!
R_tr_e_inv = []
R_tr_l_inv = []

cov_tr_e_raw = []
cov_tr_l_raw = []

for i, (f_low, f_high) in enumerate(bands):
    X_e_b = _filter_band(X_tr_early, f_low, f_high)
    X_l_b = _filter_band(X_tr_late, f_low, f_high)
    
    cov_e = cov_e_est[i].fit_transform(X_e_b)
    cov_l = cov_l_est[i].fit_transform(X_l_b)
    
    # Baseline center of training subjects
    ref_e = np.mean(cov_e, axis=0)
    ref_l = np.mean(cov_l, axis=0)
    
    inv_e = _invsqrtm(ref_e)
    inv_l = _invsqrtm(ref_l)
    
    R_tr_e_inv.append(inv_e)
    R_tr_l_inv.append(inv_l)
    
    # Pre-whiten training data to I: C_tilde = R^{-1/2} C R^{-1/2}
    cov_e_aligned = np.array([inv_e @ c @ inv_e for c in cov_e])
    cov_l_aligned = np.array([inv_l @ c @ inv_l for c in cov_l])
    
    cov_tr_e_raw.append(cov_e_aligned)
    cov_tr_l_raw.append(cov_l_aligned)

ts_e_ra = [TangentSpace(metric="riemann") for _ in bands]
ts_l_ra = [TangentSpace(metric="riemann") for _ in bands]

tokens_tr_ra = []
delta_tr_ra = []

for i in range(n_bands):
    corr_e, diag_e = _cov_to_corr(cov_tr_e_raw[i])
    conn_e = ts_e_ra[i].fit_transform(corr_e)
    
    corr_l, diag_l = _cov_to_corr(cov_tr_l_raw[i])
    conn_l = ts_l_ra[i].fit_transform(corr_l)
    
    p_e = np.log(np.maximum(diag_e, 1e-12))
    p_l = np.log(np.maximum(diag_l, 1e-12))
    p_delta = p_l - p_e
    
    tokens_tr_ra.append(np.concatenate([p_e, conn_e], axis=1))
    tokens_tr_ra.append(np.concatenate([p_l, conn_l], axis=1))
    delta_tr_ra.append(p_delta)

flat_tr_ra = np.concatenate(tokens_tr_ra + delta_tr_ra, axis=1)
scaler_ra = StandardScaler()
flat_tr_ra = scaler_ra.fit_transform(flat_tr_ra)

clf_ra = RidgeClassifier(alpha=150.0)
clf_ra.fit(flat_tr_ra, y_train)

# Step B: Streaming Online Inference (Simulating causal predict_streams)
# Causal update: test windows arrive one by one!
running_sum_e = [np.zeros((41, 41)) for _ in range(n_bands)]
running_sum_l = [np.zeros((41, 41)) for _ in range(n_bands)]

preds_adaptive = []

for t in range(len(X_test)):
    w_e = X_te_early[t:t+1]
    w_l = X_te_late[t:t+1]
    
    tokens_curr = []
    delta_curr = []
    
    for i, (f_low, f_high) in enumerate(bands):
        X_e_b = _filter_band(w_e, f_low, f_high)
        X_l_b = _filter_band(w_l, f_low, f_high)
        
        c_e = cov_e_est[i].transform(X_e_b)[0]
        c_l = cov_l_est[i].transform(X_l_b)[0]
        
        # Online running baseline accumulation
        running_sum_e[i] += c_e
        running_sum_l[i] += c_l
        
        ref_e_test = running_sum_e[i] / (t + 1)
        ref_l_test = running_sum_l[i] / (t + 1)
        
        inv_e_test = _invsqrtm(ref_e_test)
        inv_l_test = _invsqrtm(ref_l_test)
        
        # Pre-whiten incoming test window to I
        c_e_aligned = (inv_e_test @ c_e @ inv_e_test)[np.newaxis, ...]
        c_l_aligned = (inv_l_test @ c_l @ inv_l_test)[np.newaxis, ...]
        
        corr_e, diag_e = _cov_to_corr(c_e_aligned)
        conn_e = ts_e_ra[i].transform(corr_e)
        
        corr_l, diag_l = _cov_to_corr(c_l_aligned)
        conn_l = ts_l_ra[i].transform(corr_l)
        
        p_e = np.log(np.maximum(diag_e, 1e-12))
        p_l = np.log(np.maximum(diag_l, 1e-12))
        p_delta = p_l - p_e
        
        tokens_curr.append(np.concatenate([p_e, conn_e], axis=1))
        tokens_curr.append(np.concatenate([p_l, conn_l], axis=1))
        delta_curr.append(p_delta)
        
    flat_curr = np.concatenate(tokens_curr + delta_curr, axis=1)
    flat_curr_scaled = scaler_ra.transform(flat_curr)
    
    p = clf_ra.predict(flat_curr_scaled)[0]
    preds_adaptive.append(p)

preds_adaptive = np.array(preds_adaptive)
acc_adaptive = balanced_accuracy_score(y_test, preds_adaptive)

print("\n" + "=" * 65)
print("FINAL BENCHMARK: FULL 12-TOKEN BACKBONE ON UNSEEN TEST STREAM")
print("=" * 65)
print(f"1. Static Champion Model (Frozen State):       {acc_static * 100:.2f}%")
print(f"2. Adaptive Champion Model (Online Alignment):  {acc_adaptive * 100:.2f}%")
print(f"Net Score Improvement:                         {(acc_adaptive - acc_static) * 100:+.2f}%")
print("=" * 65)

