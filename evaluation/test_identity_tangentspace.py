"""Test TangentSpace with True Identity Reference (Cr = I_41).

Hypothesis:
Using Cr = I_41 ensures that both training subjects and test subjects project to the tangent space
centered at the exact same point (Identity). This eliminates the artificial subject-specific offset
v_subject that causes the hyperplane tilt (e.g. Subject 10 Math bias, Subject 4 Word bias).
"""

import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task
from benchmark_utils.data import to_numpy
from experiment_unified_41ch_sequence import Unified41ChSequencePipeline, _invsqrtm, _cov_to_corr

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

ch_names = "C2,CP2,P2,Pz,CPz,Cz,C1,CP1,P1,P3,PO7,P5,TP7,CP5,CP3,C3,PO8,P6,P4,TP8,CP6,CP4,C6,C4,FC6,FC4,FC2,F6,F4,F2,AF4,AF3,F5,F3,F1,Fz,FC5,FC3,FC1,FCz,C5".split(",")
pipeline = Unified41ChSequencePipeline(ch_names=ch_names, sfreq=120.0, prior_N0=25)

# Fit pipeline
print("Fitting training pipeline with Cr = I_41...")
X_all_tr, y_all_tr = [], []
for X, y, _ in loaders["train"]:
    X_all_tr.append(to_numpy(X))
    y_np = to_numpy(y)
    y_all_tr.append(y_np.argmax(axis=-1) if y_np.ndim > 1 else y_np)

X_all_tr = np.concatenate(X_all_tr, axis=0)
y_all_tr = np.concatenate(y_all_tr, axis=0)

features_tr = []
tau = 10

sos_bb = butter(4, [1.0, 40.0], btype="bandpass", fs=pipeline.sfreq, output="sos")
X_bb = sosfiltfilt(sos_bb, X_all_tr, axis=-1)
sf_bb = np.mean(X_bb[:, pipeline.f_idx], axis=1)
sp_bb = np.mean(X_bb[:, pipeline.p_idx], axis=1)
sf_bb = (sf_bb - sf_bb.mean(axis=1, keepdims=True)) / (sf_bb.std(axis=1, keepdims=True) + 1e-12)
sp_bb = (sp_bb - sp_bb.mean(axis=1, keepdims=True)) / (sp_bb.std(axis=1, keepdims=True) + 1e-12)
flow_bb = np.mean(sf_bb[:, :-tau] * sp_bb[:, tau:], axis=1, keepdims=True) - \
          np.mean(sp_bb[:, :-tau] * sf_bb[:, tau:], axis=1, keepdims=True)

for b, (fl, fh) in enumerate(pipeline.bands):
    X_filt = pipeline._filter_band(X_all_tr, fl, fh)
    X_e = X_filt[..., :240]
    X_l = X_filt[..., 240:]

    cov_e = pipeline.cov_early[b].fit_transform(X_e)
    cov_l = pipeline.cov_late[b].fit_transform(X_l)

    ref_e = np.mean(cov_e, axis=0)
    ref_l = np.mean(cov_l, axis=0)
    pipeline.ref_e_train[b] = ref_e
    pipeline.ref_l_train[b] = ref_l

    inv_e = _invsqrtm(ref_e)
    inv_l = _invsqrtm(ref_l)

    cov_e_al = np.array([inv_e @ c @ inv_e for c in cov_e])
    cov_l_al = np.array([inv_l @ c @ inv_l for c in cov_l])

    corr_e, diag_e = _cov_to_corr(cov_e_al)
    corr_l, diag_l = _cov_to_corr(cov_l_al)

    # Force reference to EXACT IDENTITY I_41
    pipeline.ts_early[b].fit(corr_e)
    pipeline.ts_early[b].reference_ = np.eye(pipeline.n_chans)

    pipeline.ts_late[b].fit(corr_l)
    pipeline.ts_late[b].reference_ = np.eye(pipeline.n_chans)

    conn_e = pipeline.ts_early[b].transform(corr_e)
    conn_l = pipeline.ts_late[b].transform(corr_l)

    p_e = np.log(np.maximum(diag_e, 1e-12))
    p_l = np.log(np.maximum(diag_l, 1e-12))
    p_macro = p_l - p_e

    p_stages = []
    for s, (t0, t1) in enumerate(pipeline.stages):
        seg = X_filt[..., t0:t1]
        p_s = np.log(np.mean(seg ** 2, axis=-1) + 1e-12)
        p_stages.append(p_s)

    p_all = np.stack(p_stages, axis=1)
    p_trial_mean = np.mean(p_all, axis=1, keepdims=True)
    p_norm = p_all - p_trial_mean

    s_01 = p_norm[:, 1] - p_norm[:, 0]
    s_12 = p_norm[:, 2] - p_norm[:, 1]
    s_23 = p_norm[:, 3] - p_norm[:, 2]
    s_net = p_norm[:, 3] - p_norm[:, 0]
    curv = s_12 - s_01

    pF_stages = np.mean(p_norm[:, :, pipeline.f_idx], axis=-1)
    pM_stages = np.mean(p_norm[:, :, pipeline.m_idx], axis=-1)
    pP_stages = np.mean(p_norm[:, :, pipeline.p_idx], axis=-1)

    cFP = pF_stages - pP_stages
    cFM = pF_stages - pM_stages
    cPM = pP_stages - pM_stages

    sFP_01 = cFP[:, 1:2] - cFP[:, 0:1]
    sFP_12 = cFP[:, 2:3] - cFP[:, 1:2]
    sFP_23 = cFP[:, 3:4] - cFP[:, 2:3]
    sFM_01 = cFM[:, 1:2] - cFM[:, 0:1]
    sFM_12 = cFM[:, 2:3] - cFM[:, 1:2]
    sPM_01 = cPM[:, 1:2] - cPM[:, 0:1]
    sPM_12 = cPM[:, 2:3] - cPM[:, 1:2]

    sf_b = np.mean(X_filt[:, pipeline.f_idx], axis=1)
    sp_b = np.mean(X_filt[:, pipeline.p_idx], axis=1)
    sf_b = (sf_b - sf_b.mean(axis=1, keepdims=True)) / (sf_b.std(axis=1, keepdims=True) + 1e-12)
    sp_b = (sp_b - sp_b.mean(axis=1, keepdims=True)) / (sp_b.std(axis=1, keepdims=True) + 1e-12)
    flow_b = np.mean(sf_b[:, :-tau] * sp_b[:, tau:], axis=1, keepdims=True) - \
             np.mean(sp_b[:, :-tau] * sf_b[:, tau:], axis=1, keepdims=True)

    features_tr.extend([
        conn_e, conn_l, p_e, p_l, p_macro,
        p_norm[:, 0], p_norm[:, 1], p_norm[:, 2], p_norm[:, 3],
        s_01, s_12, s_23, s_net, curv,
        cFP, cFM, cPM,
        sFP_01, sFP_12, sFP_23,
        sFM_01, sFM_12,
        sPM_01, sPM_12,
        flow_b
    ])

features_tr.append(flow_bb)
X_tr = np.concatenate(features_tr, axis=1)
scaler = StandardScaler().fit(X_tr)
ridge = RidgeClassifierCV(alphas=[50, 100, 250, 500, 1000, 2500, 5000, 10000]).fit(scaler.transform(X_tr), y_all_tr)
print(f"Ridge selected alpha: {ridge.alpha_}")

# Evaluate on test streams
test_loader = loaders["test"]
X_all_test = []
y_all_test = []
stream_ids = []

for X, y, info in test_loader:
    X_all_test.append(X)
    y_np = to_numpy(y)
    y_all_test.append(y_np.argmax(-1) if y_np.ndim > 1 else y_np)
    stream_ids.append(info["stream_id"][0] if isinstance(info["stream_id"], list) else info["stream_id"])

y_all_test = np.concatenate(y_all_test, axis=0).reshape(-1)
stream_ids = np.array(stream_ids).reshape(-1)
unique_streams = np.unique(stream_ids)

# Test evaluation across priors
for n0 in [15, 25, 40]:
    pipeline.prior_N0 = n0
    pipeline.reset_state()
    cur_stream = None
    preds = []
    margins = []

    for X, sid in zip(X_all_test, stream_ids):
        if sid != cur_stream:
            cur_stream = sid
            pipeline.reset_state()

        X_np = to_numpy(X)
        if X_np.ndim == 2:
            X_np = X_np[np.newaxis, ...]

        pipeline.n_test_windows += 1
        feat_curr = []

        sos_bb = butter(4, [1.0, 40.0], btype="bandpass", fs=pipeline.sfreq, output="sos")
        X_bb = sosfiltfilt(sos_bb, X_np, axis=-1)
        sf_bb = np.mean(X_bb[0, pipeline.f_idx], axis=0)
        sp_bb = np.mean(X_bb[0, pipeline.p_idx], axis=0)
        sf_bb = (sf_bb - sf_bb.mean()) / (sf_bb.std() + 1e-12)
        sp_bb = (sp_bb - sp_bb.mean()) / (sp_bb.std() + 1e-12)
        flow_bb_val = np.array([[np.mean(sf_bb[:-tau] * sp_bb[tau:]) - np.mean(sp_bb[:-tau] * sf_bb[tau:])]])

        for b, (fl, fh) in enumerate(pipeline.bands):
            X_filt = pipeline._filter_band(X_np, fl, fh)
            w_e = X_filt[..., :240]
            w_l = X_filt[..., 240:]

            c_e = pipeline.cov_early[b].transform(w_e)[0]
            c_l = pipeline.cov_late[b].transform(w_l)[0]
            pipeline.running_cov_e[b] += c_e
            pipeline.running_cov_l[b] += c_l

            ref_e_test = pipeline.running_cov_e[b] / pipeline.n_test_windows
            ref_l_test = pipeline.running_cov_l[b] / pipeline.n_test_windows

            inv_e = _invsqrtm(ref_e_test)
            inv_l = _invsqrtm(ref_l_test)

            c_e_al = (inv_e @ c_e @ inv_e)[np.newaxis, ...]
            c_l_al = (inv_l @ c_l @ inv_l)[np.newaxis, ...]

            corr_e, diag_e = _cov_to_corr(c_e_al)
            corr_l, diag_l = _cov_to_corr(c_l_al)

            # Transform with Cr = I_41
            conn_e = pipeline.ts_early[b].transform(corr_e)
            conn_l = pipeline.ts_late[b].transform(corr_l)

            p_e = np.log(np.maximum(diag_e, 1e-12))
            p_l = np.log(np.maximum(diag_l, 1e-12))
            p_macro = p_l - p_e

            p_stages = []
            for s, (t0, t1) in enumerate(pipeline.stages):
                seg = X_filt[..., t0:t1]
                p_s = np.log(np.mean(seg ** 2, axis=-1) + 1e-12)
                p_stages.append(p_s)

            p_all = np.stack(p_stages, axis=1)
            p_trial_mean = np.mean(p_all, axis=1, keepdims=True)
            p_norm = p_all - p_trial_mean

            s_01 = p_norm[:, 1] - p_norm[:, 0]
            s_12 = p_norm[:, 2] - p_norm[:, 1]
            s_23 = p_norm[:, 3] - p_norm[:, 2]
            s_net = p_norm[:, 3] - p_norm[:, 0]
            curv = s_12 - s_01

            pF_stages = np.mean(p_norm[:, :, pipeline.f_idx], axis=-1)
            pM_stages = np.mean(p_norm[:, :, pipeline.m_idx], axis=-1)
            pP_stages = np.mean(p_norm[:, :, pipeline.p_idx], axis=-1)

            cFP = pF_stages - pP_stages
            cFM = pF_stages - pM_stages
            cPM = pP_stages - pM_stages

            sFP_01 = cFP[:, 1:2] - cFP[:, 0:1]
            sFP_12 = cFP[:, 2:3] - cFP[:, 1:2]
            sFP_23 = cFP[:, 3:4] - cFP[:, 2:3]
            sFM_01 = cFM[:, 1:2] - cFM[:, 0:1]
            sFM_12 = cFM[:, 2:3] - cFM[:, 1:2]
            sPM_01 = cPM[:, 1:2] - cPM[:, 0:1]
            sPM_12 = cPM[:, 2:3] - cPM[:, 1:2]

            sf_b = np.mean(X_filt[0, pipeline.f_idx], axis=0)
            sp_b = np.mean(X_filt[0, pipeline.p_idx], axis=0)
            sf_b = (sf_b - sf_b.mean()) / (sf_b.std() + 1e-12)
            sp_b = (sp_b - sp_b.mean()) / (sp_b.std() + 1e-12)
            flow_b_val = np.array([[np.mean(sf_b[:-tau] * sp_b[tau:]) - np.mean(sp_b[:-tau] * sf_b[tau:])]])

            feat_curr.extend([
                conn_e, conn_l, p_e, p_l, p_macro,
                p_norm[:, 0], p_norm[:, 1], p_norm[:, 2], p_norm[:, 3],
                s_01, s_12, s_23, s_net, curv,
                cFP, cFM, cPM,
                sFP_01, sFP_12, sFP_23,
                sFM_01, sFM_12,
                sPM_01, sPM_12,
                flow_b_val
            ])

        feat_curr.append(flow_bb_val)
        X_vec = np.concatenate(feat_curr, axis=1)
        m = ridge.decision_function(scaler.transform(X_vec))[0]
        margins.append(m)
        preds.append(m.argmax())

    preds = np.array(preds)
    margins = np.array(margins)
    acc = balanced_accuracy_score(y_all_test, preds) * 100.0
    print(f"\n• N0 = {n0:2d} -> Identity-Referenced Balanced Acc: {acc:.2f}%")
    for sid in unique_streams:
        mask = stream_ids == sid
        s_acc = balanced_accuracy_score(y_all_test[mask], preds[mask]) * 100.0
        m_mean = margins[mask].mean(axis=0)
        print(f"   Stream {sid:2d}: BalAcc: {s_acc:5.2f}% | Mean Margins: [{m_mean[0]:.3f}, {m_mean[1]:.3f}, {m_mean[2]:.3f}]")

