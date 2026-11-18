"""Test Causal Margin Centering on Unified 41-Channel Pipeline.

Hypothesis:
Subject-specific resting-state biases cause persistent margin offsets (e.g. Subject 10 biased toward Math,
Subject 4 biased toward Word).
Tracking and centering the running mean of decision margins causally removes this subject offset,
leveling the playing field so every trial is judged purely on its task-specific modulation.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task
from benchmark_utils.data import to_numpy
from experiment_unified_41ch_sequence import Unified41ChSequencePipeline

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

pipeline = Unified41ChSequencePipeline(
    ch_names=ch_names,
    sfreq=120.0,
    prior_N0=25,
    w_ridge=0.55,
    w_logit=0.45,
)
pipeline.fit(loaders["train"])

# Extract raw margins and ground truth across all streams
classes = [0, 1, 2]
class_names = ["Motor Imagery (0)", "Mental Sub (1)", "Word Gen (2)"]

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

print(f"Total test trials: {len(y_all_test)} across streams {unique_streams}")

# We will collect raw Ridge decision margins for every trial
raw_margins_list = []
pipeline.reset_state()
cur_stream = None

for i, (X, sid) in enumerate(zip(X_all_test, stream_ids)):
    if sid != cur_stream:
        cur_stream = sid
        pipeline.reset_state()

    # Call predict_stream up to margin extraction
    X_np = to_numpy(X)
    if X_np.ndim == 2:
        X_np = X_np[np.newaxis, ...]

    pipeline.n_test_windows += 1
    features_curr = []
    tau = 10

    # Broadband directed flow
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

        inv_e = pipeline.cov_early[b].estimator.__class__
        from experiment_unified_41ch_sequence import _invsqrtm, _cov_to_corr
        inv_e = _invsqrtm(ref_e_test)
        inv_l = _invsqrtm(ref_l_test)

        c_e_al = (inv_e @ c_e @ inv_e)[np.newaxis, ...]
        c_l_al = (inv_l @ c_l @ inv_l)[np.newaxis, ...]

        corr_e, diag_e = _cov_to_corr(c_e_al)
        corr_l, diag_l = _cov_to_corr(c_l_al)

        conn_e = pipeline.ts_early[b].transform(corr_e)
        conn_l = pipeline.ts_late[b].transform(corr_l)

        p_e = np.log(np.maximum(diag_e, 1e-12))
        p_l = np.log(np.maximum(diag_l, 1e-12))
        p_macro_delta = p_l - p_e

        features_curr.extend([conn_e, conn_l, p_e, p_l, p_macro_delta])

        # Sequence dynamics
        p_stages = []
        for s, (t_start, t_end) in enumerate(pipeline.stages):
            seg = X_filt[..., t_start:t_end]
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

        features_curr.extend([
            p_norm[:, 0], p_norm[:, 1], p_norm[:, 2], p_norm[:, 3],
            s_01, s_12, s_23, s_net, curv
        ])

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

        features_curr.extend([
            cFP, cFM, cPM,
            sFP_01, sFP_12, sFP_23,
            sFM_01, sFM_12,
            sPM_01, sPM_12,
        ])

        sf_b = np.mean(X_filt[0, pipeline.f_idx], axis=0)
        sp_b = np.mean(X_filt[0, pipeline.p_idx], axis=0)
        sf_b = (sf_b - sf_b.mean()) / (sf_b.std() + 1e-12)
        sp_b = (sp_b - sp_b.mean()) / (sp_b.std() + 1e-12)
        flow_b_val = np.array([[np.mean(sf_b[:-tau] * sp_b[tau:]) - np.mean(sp_b[:-tau] * sf_b[tau:])]])
        features_curr.append(flow_b_val)

    features_curr.append(flow_bb_val)

    X_flat = np.concatenate(features_curr, axis=1)
    X_scaled = pipeline.scaler.transform(X_flat)
    m = pipeline.ridge.decision_function(X_scaled)[0]
    raw_margins_list.append(m)

raw_margins = np.array(raw_margins_list)  # (755, 3)

print("\n" + "=" * 80)
print("MEAN RAW MARGINS PER STREAM (SUBJECT BIAS AUDIT)")
print("=" * 80)
print(f"{'Stream ID':<12} | {'Mean Motor Margin':<20} | {'Mean Math Margin':<20} | {'Mean Word Margin':<20}")
print("-" * 80)
for sid in unique_streams:
    mask = stream_ids == sid
    m_mean = raw_margins[mask].mean(axis=0)
    print(f"Stream {sid:<5d} | {m_mean[0]:20.4f} | {m_mean[1]:20.4f} | {m_mean[2]:20.4f}")

# Baseline raw accuracy
baseline_preds = raw_margins.argmax(axis=1)
base_bal_acc = balanced_accuracy_score(y_all_test, baseline_preds) * 100.0
print(f"\nBaseline Pooled Balanced Accuracy (without centering): {base_bal_acc:.2f}%")

# Now evaluate CAUSAL margin centering:
# As each stream runs, we update running_mean_margin:
# mu(t) = ((t-1)*mu(t-1) + m(t)) / t
# centered_m(t) = m(t) - mu(t-1)   [causal: uses past trials only!]
print("\n" + "=" * 80)
print("TESTING CAUSAL MARGIN CENTERING WITH EXPONENTIAL MOVING AVERAGE (EMA)")
print("=" * 80)

for tau_ema in [10, 20, 30, 50, 100, 999999]:  # 999999 = cumulative mean
    causal_preds = []
    for sid in unique_streams:
        mask = stream_ids == sid
        s_margins = raw_margins[mask]
        s_preds = []
        running_mu = np.zeros(3)
        alpha = 1.0 / tau_ema if tau_ema < 1000 else None

        for t, m in enumerate(s_margins):
            if t == 0:
                s_preds.append(m.argmax())
                running_mu = m.copy()
            else:
                # Causal centered prediction using past baseline
                centered_m = m - running_mu
                s_preds.append(centered_m.argmax())
                if alpha is not None:
                    running_mu = (1 - alpha) * running_mu + alpha * m
                else:
                    running_mu = (t * running_mu + m) / (t + 1)

        causal_preds.extend(s_preds)

    causal_preds = np.array(causal_preds)
    causal_bal_acc = balanced_accuracy_score(y_all_test, causal_preds) * 100.0
    label = f"EMA tau={tau_ema}" if tau_ema < 1000 else "Cumulative Mean"
    print(f"• {label:<22} -> Pooled Balanced Acc: {causal_bal_acc:5.2f}% (Δ = {causal_bal_acc - base_bal_acc:+5.2f}%)")

