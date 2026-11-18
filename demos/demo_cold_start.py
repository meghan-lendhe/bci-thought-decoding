"""Investigating the Cold-Start Trap in Online Alignment.

Shows why naive online alignment dropped by -1.54%:
1. At t=0, running mean has only 1 trial -> C_test^{-1/2} @ C_0 @ C_test^{-1/2} = IDENTITY MATRIX (all info wiped out!).
2. Testing the standard solution: Bayesian Prior Regularization (initializing the test running mean with the training center!).
"""

import sys
from pathlib import Path
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

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
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

X_test_list, y_test_list = [], []
for X_b, y_b, info in loaders["test"]:
    X_test_list.append(X_b.cpu().numpy()[0])
    y_test_list.append(int(y_b.cpu().numpy()[0]))
X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

bands = [
    (4.0, 8.0), (8.0, 11.0), (11.0, 14.0),
    (14.0, 20.0), (20.0, 28.0), (28.0, 40.0)
]
n_bands = len(bands)

def _filter_band(X, f_low, f_high):
    sos = butter(4, [f_low, f_high], btype="bandpass", fs=120.0, output="sos")
    return sosfiltfilt(sos, X, axis=-1)

def _cov_to_corr(covs):
    diags = np.diagonal(covs, axis1=-2, axis2=-1)
    stds = np.sqrt(np.maximum(diags, 1e-12))[..., np.newaxis]
    corrs = covs / (stds @ stds.swapaxes(-2, -1))
    for i in range(len(corrs)):
        np.fill_diagonal(corrs[i], 1.0)
    return corrs, diags

def _invsqrtm(C, eps=1e-5):
    # Add regularizing shrinkage
    trace = np.trace(C) / len(C)
    C_reg = 0.95 * C + 0.05 * trace * np.eye(len(C))
    eigvals, eigvecs = np.linalg.eigh(C_reg)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

split_idx = X_train.shape[-1] // 2
X_tr_early, X_tr_late = X_train[..., :split_idx], X_train[..., split_idx:]
X_te_early, X_te_late = X_test[..., :split_idx], X_test[..., split_idx:]

cov_e_est = [Covariances(estimator="lwf") for _ in bands]
cov_l_est = [Covariances(estimator="lwf") for _ in bands]

# Training Step
ref_e_train = []
ref_l_train = []
R_tr_e_inv = []
R_tr_l_inv = []
cov_tr_e_raw = []
cov_tr_l_raw = []

for i, (f_low, f_high) in enumerate(bands):
    X_e_b = _filter_band(X_tr_early, f_low, f_high)
    X_l_b = _filter_band(X_tr_late, f_low, f_high)
    
    cov_e = cov_e_est[i].fit_transform(X_e_b)
    cov_l = cov_l_est[i].fit_transform(X_l_b)
    
    ref_e = np.mean(cov_e, axis=0)
    ref_l = np.mean(cov_l, axis=0)
    ref_e_train.append(ref_e)
    ref_l_train.append(ref_l)
    
    inv_e = _invsqrtm(ref_e)
    inv_l = _invsqrtm(ref_l)
    R_tr_e_inv.append(inv_e)
    R_tr_l_inv.append(inv_l)
    
    cov_tr_e_raw.append(np.array([inv_e @ c @ inv_e for c in cov_e]))
    cov_tr_l_raw.append(np.array([inv_l @ c @ inv_l for c in cov_l]))

ts_e_ra = [TangentSpace(metric="riemann") for _ in bands]
ts_l_ra = [TangentSpace(metric="riemann") for _ in bands]

tokens_tr = []
delta_tr = []
for i in range(n_bands):
    corr_e, diag_e = _cov_to_corr(cov_tr_e_raw[i])
    conn_e = ts_e_ra[i].fit_transform(corr_e)
    corr_l, diag_l = _cov_to_corr(cov_tr_l_raw[i])
    conn_l = ts_l_ra[i].fit_transform(corr_l)
    
    p_e = np.log(np.maximum(diag_e, 1e-12))
    p_l = np.log(np.maximum(diag_l, 1e-12))
    tokens_tr.append(np.concatenate([p_e, conn_e], axis=1))
    tokens_tr.append(np.concatenate([p_l, conn_l], axis=1))
    delta_tr.append(p_l - p_e)

flat_tr = np.concatenate(tokens_tr + delta_tr, axis=1)
scaler = StandardScaler()
flat_tr = scaler.fit_transform(flat_tr)

clf = RidgeClassifier(alpha=150.0)
clf.fit(flat_tr, y_train)

# Test with various Prior Weights (N_prior)
# N_prior = 0: Naive (what we just did)
# N_prior = 10, 20, 30: Warm start using training center
for N_prior in [0, 5, 10, 20]:
    running_sum_e = [N_prior * ref_e_train[i].copy() for i in range(n_bands)]
    running_sum_l = [N_prior * ref_l_train[i].copy() for i in range(n_bands)]
    
    preds = []
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
            
            running_sum_e[i] += c_e
            running_sum_l[i] += c_l
            
            # Smooth Bayesian update
            ref_e_test = running_sum_e[i] / (N_prior + t + 1)
            ref_l_test = running_sum_l[i] / (N_prior + t + 1)
            
            inv_e = _invsqrtm(ref_e_test)
            inv_l = _invsqrtm(ref_l_test)
            
            c_e_al = (inv_e @ c_e @ inv_e)[np.newaxis, ...]
            c_l_al = (inv_l @ c_l @ inv_l)[np.newaxis, ...]
            
            corr_e, diag_e = _cov_to_corr(c_e_al)
            corr_l, diag_l = _cov_to_corr(c_l_al)
            
            conn_e = ts_e_ra[i].transform(corr_e)
            conn_l = ts_l_ra[i].transform(corr_l)
            
            p_e = np.log(np.maximum(diag_e, 1e-12))
            p_l = np.log(np.maximum(diag_l, 1e-12))
            tokens_curr.append(np.concatenate([p_e, conn_e], axis=1))
            tokens_curr.append(np.concatenate([p_l, conn_l], axis=1))
            delta_curr.append(p_l - p_e)
            
        flat_curr = scaler.transform(np.concatenate(tokens_curr + delta_curr, axis=1))
        preds.append(clf.predict(flat_curr)[0])
        
    acc = balanced_accuracy_score(y_test, preds)
    print(f"Prior Weight N_prior = {N_prior:2d} -> Test Stream Balanced Accuracy: {acc * 100:.2f}%")

