"""Adaptive Guarded Hierarchical Solver for 3-Class BCI (Solution A: Direct Raw Formulation).

Key Capabilities:
1. Full 13,426-Dimensional Riemannian + Sequence Representation:
   - 6 Frequency Bands: Theta (4-8Hz), Low Alpha (8-11Hz), Mu (11-14Hz), Low Beta (14-20Hz), High Beta (20-28Hz), Low Gamma (28-40Hz).
   - Early (0-2s) and Late (2-4s) online Riemannian tangent space projections (2 x 861 = 1,722 dims per band x 6 = 10,332 dims).
   - 4-Stage Power Dynamics (0-1s, 1-2s, 2-3s, 3-4s) + Net Progression + Curvature across 41 channels.
   - Regional Subsystem Contrasts (Frontal, Motor, Parietal) and Inter-stage Flow Slopes.
   - Directional Lagged Information Flow (tau=10).
   - Low-Beta Lateralization Contrast (TP8 vs TP7).

2. Bayesian Warm-Start Online Riemannian Adaptation:
   - Running covariance accumulators initialized with population prior (prior_N0 = 25).
   - Per-stream reset_state() contract.

3. Solution A (Direct Raw Guarded Hierarchical Decision Architecture):
   - OvO Binary Specialists: 0v1 (Motor vs Math), 0v2 (Motor vs Word), 1v2 (Math vs Word).
   - Operates directly on the model's natural raw margins without division by noisy or mismatched standard deviations.
   - Motor Gate with Causal Bayesian Damping:
     ms_raw = -(raw01 + raw02) / 2.0
     th_mot = (sum ms_raw) / (M_mot + t) + base_th_raw (0.01)
   - Cognitive Specialist with Causal Damping and Lateralization Contrast:
     th_cog = (sum raw12) / (M_cog + t)
     eff_12 = (raw12 - th_cog) - (w_lat_raw * 0.16) * z_lat
   - Cognitive Confidence Guard:
     Prevents borderline Motor false alarms from overriding high-confidence Math/Word trials:
     Predict 0 iff (ms_raw > th_mot) AND (|eff_12| < guard_raw, 0.60).
"""

import copy
from pathlib import Path
import joblib
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


def _invsqrtm(C, eps=1e-5):
    trace = np.trace(C) / len(C)
    C_reg = 0.95 * C + 0.05 * trace * np.eye(len(C))
    eigvals, eigvecs = np.linalg.eigh(C_reg)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _cov_to_corr(covs):
    diags = np.diagonal(covs, axis1=-2, axis2=-1)
    stds = np.sqrt(np.maximum(diags, 1e-12))[..., np.newaxis]
    corrs = covs / (stds @ stds.swapaxes(-2, -1))
    for i in range(len(corrs)):
        np.fill_diagonal(corrs[i], 1.0)
    return corrs, diags


class AdaptiveGuardedHierarchicalModel:
    """End-to-End Streaming Model with Causal Riemannian Alignment and Guarded Hierarchical Gating."""

    def __init__(
        self,
        n_chans=41,
        n_classes=3,
        sfreq=120.0,
        prior_N0=25,
        base_th_raw=0.01,
        guard_raw=0.60,
        w_lat_raw=2.0,
        M_mot=10,
        M_cog=10,
        M_lat=15,
        record_trace=False,
        device="cpu",
    ):
        self.n_chans = n_chans
        self.n_classes = n_classes
        self.sfreq = sfreq
        self.prior_N0 = prior_N0
        self.base_th_raw = base_th_raw
        self.guard_raw = guard_raw
        self.w_lat_raw = w_lat_raw
        self.M_mot = M_mot
        self.M_cog = M_cog
        self.M_lat = M_lat
        self.record_trace = record_trace
        self.device = device

        self.bands = [
            (4.0, 8.0),
            (8.0, 11.0),
            (11.0, 14.0),
            (14.0, 20.0),
            (20.0, 28.0),
            (28.0, 40.0),
        ]
        self.stages = [
            (24, 120),
            (120, 240),
            (240, 360),
            (360, 480),
        ]
        self.tau = 10

        self.ch_names = "C2,CP2,P2,Pz,CPz,Cz,C1,CP1,P1,P3,PO7,P5,TP7,CP5,CP3,C3,PO8,P6,P4,TP8,CP6,CP4,C6,C4,FC6,FC4,FC2,F6,F4,F2,AF4,AF3,F5,F3,F1,Fz,FC5,FC3,FC1,FCz,C5".split(",")
        self.f_chs = ['FC6', 'FC4', 'FC2', 'F6', 'F4', 'F2', 'AF4', 'AF3', 'F5', 'F3', 'F1', 'Fz', 'FC5', 'FC3', 'FC1', 'FCz']
        self.m_chs = ['C2', 'Cz', 'C1', 'C3', 'C6', 'C4', 'C5']
        self.p_chs = ['CP2', 'P2', 'Pz', 'CPz', 'CP1', 'P1', 'P3', 'PO7', 'P5', 'CP5', 'CP3', 'PO8', 'P6', 'P4', 'CP6', 'CP4']

        self.f_idx = [self.ch_names.index(ch) for ch in self.f_chs if ch in self.ch_names]
        self.m_idx = [self.ch_names.index(ch) for ch in self.m_chs if ch in self.ch_names]
        self.p_idx = [self.ch_names.index(ch) for ch in self.p_chs if ch in self.ch_names]

        self.idx_tp8 = self.ch_names.index("TP8")
        self.idx_fc2 = self.ch_names.index("FC2")
        self.idx_fc1 = self.ch_names.index("FC1")
        self.idx_tp7 = self.ch_names.index("TP7")
        self.idx_cp5 = self.ch_names.index("CP5")
        self.idx_f5  = self.ch_names.index("F5")
        self.idx_fc5 = self.ch_names.index("FC5")

        self.cov_early = [Covariances(estimator="lwf") for _ in self.bands]
        self.cov_late  = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_early  = [TangentSpace(metric="riemann") for _ in self.bands]
        self.ts_late   = [TangentSpace(metric="riemann") for _ in self.bands]

        self.scaler = StandardScaler()
        alphas = np.logspace(-2, 5, 20)
        self.ridge_01 = RidgeClassifierCV(alphas=alphas, class_weight="balanced")
        self.ridge_02 = RidgeClassifierCV(alphas=alphas, class_weight="balanced")
        self.ridge_12 = RidgeClassifierCV(alphas=alphas, class_weight="balanced")

        self.ref_e_train = None
        self.ref_l_train = None
        self.mu_lat = 0.0
        self.s_lat = 1.0
        self.fitted = False

        self.reset_state()

    def _filter_band(self, X, fl, fh):
        sos = butter(4, [fl, fh], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)

    def reset_state(self):
        """Reset online running statistics at the start of a new stream."""
        if self.ref_e_train is not None:
            self.running_cov_e = [self.prior_N0 * ref.copy() for ref in self.ref_e_train]
            self.running_cov_l = [self.prior_N0 * ref.copy() for ref in self.ref_l_train]
        else:
            self.running_cov_e = None
            self.running_cov_l = None

        self.n_test_windows = self.prior_N0
        self.r_mot = 0.0
        self.n_mot = 0
        self.r_cog = 0.0
        self.n_cog = 0
        self.r_lat = 0.0
        self.n_lat = 0

        if getattr(self, "record_trace", False):
            if not hasattr(self, "trace_by_stream"):
                self.trace_by_stream = []
            self.current_stream_trace = []
            self.trace_by_stream.append(self.current_stream_trace)

    def fit(self, train_loader):
        print("Fitting AdaptiveGuardedHierarchicalModel (Solution A: Direct Raw Margin)...")
        X_all_tr, y_all_tr = [], []
        for X, y, _ in train_loader:
            X_np = to_numpy(X)
            if X_np.ndim == 2:
                X_np = X_np[np.newaxis, ...]
            X_all_tr.append(X_np)
            y_np = to_numpy(y)
            y_all_tr.append(int(y_np.argmax(-1)) if y_np.ndim > 1 else int(y_np.flatten()[0]))
        X_all_tr = np.concatenate(X_all_tr, axis=0)
        y_all_tr = np.array(y_all_tr).flatten()

        print(f"Loaded {len(X_all_tr)} train trials. Extracting 13,426 features...")
        feat_tr_list = []
        self.ref_e_train, self.ref_l_train = [], []

        sos_bb = butter(4, [1.0, 40.0], btype="bandpass", fs=self.sfreq, output="sos")
        X_bb_tr = sosfiltfilt(sos_bb, X_all_tr, axis=-1)
        sf_bb_tr = np.mean(X_bb_tr[:, self.f_idx], axis=1)
        sp_bb_tr = np.mean(X_bb_tr[:, self.p_idx], axis=1)
        sf_bb_tr = (sf_bb_tr - sf_bb_tr.mean(axis=-1, keepdims=True)) / (sf_bb_tr.std(axis=-1, keepdims=True) + 1e-12)
        sp_bb_tr = (sp_bb_tr - sp_bb_tr.mean(axis=-1, keepdims=True)) / (sp_bb_tr.std(axis=-1, keepdims=True) + 1e-12)
        flow_bb_tr = np.mean(sf_bb_tr[:, :-self.tau] * sp_bb_tr[:, self.tau:], axis=-1, keepdims=True) - \
                     np.mean(sp_bb_tr[:, :-self.tau] * sf_bb_tr[:, self.tau:], axis=-1, keepdims=True)

        lat_lb_train = []

        for b, (fl, fh) in enumerate(self.bands):
            X_filt_tr = self._filter_band(X_all_tr, fl, fh)
            w_early = X_filt_tr[..., :240]
            w_late  = X_filt_tr[..., 240:]

            c_e = self.cov_early[b].fit_transform(w_early)
            c_l = self.cov_late[b].fit_transform(w_late)

            r_e = np.mean(c_e, axis=0)
            r_l = np.mean(c_l, axis=0)
            self.ref_e_train.append(r_e)
            self.ref_l_train.append(r_l)

            inv_e = _invsqrtm(r_e)
            inv_l = _invsqrtm(r_l)

            c_e_al = inv_e @ c_e @ inv_e
            c_l_al = inv_l @ c_l @ inv_l

            corr_e, diag_e = _cov_to_corr(c_e_al)
            corr_l, diag_l = _cov_to_corr(c_l_al)

            conn_e = self.ts_early[b].fit_transform(corr_e)
            conn_l = self.ts_late[b].fit_transform(corr_l)

            p_e = np.log(np.maximum(diag_e, 1e-12))
            p_l = np.log(np.maximum(diag_l, 1e-12))
            p_macro = p_l - p_e

            p_stages = []
            for s, (t0, t1) in enumerate(self.stages):
                seg = X_filt_tr[..., t0:t1]
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

            pF_stages = np.mean(p_norm[:, :, self.f_idx], axis=-1)
            pM_stages = np.mean(p_norm[:, :, self.m_idx], axis=-1)
            pP_stages = np.mean(p_norm[:, :, self.p_idx], axis=-1)

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

            sf_b = np.mean(X_filt_tr[:, self.f_idx], axis=1)
            sp_b = np.mean(X_filt_tr[:, self.p_idx], axis=1)
            sf_b = (sf_b - sf_b.mean(axis=-1, keepdims=True)) / (sf_b.std(axis=-1, keepdims=True) + 1e-12)
            sp_b = (sp_b - sp_b.mean(axis=-1, keepdims=True)) / (sp_b.std(axis=-1, keepdims=True) + 1e-12)
            flow_b = np.mean(sf_b[:, :-self.tau] * sp_b[:, self.tau:], axis=-1, keepdims=True) - \
                     np.mean(sp_b[:, :-self.tau] * sf_b[:, self.tau:], axis=-1, keepdims=True)

            feat_tr_list.extend([
                conn_e, conn_l, p_e, p_l, p_macro,
                p_norm[:, 0], p_norm[:, 1], p_norm[:, 2], p_norm[:, 3],
                s_01, s_12, s_23, s_net, curv,
                cFP, cFM, cPM,
                sFP_01, sFP_12, sFP_23,
                sFM_01, sFM_12,
                sPM_01, sPM_12,
                flow_b
            ])

            if b == 3:  # Low Beta Lateralization
                for i_tr in range(len(X_filt_tr)):
                    c_tr = np.corrcoef(X_filt_tr[i_tr, :, 120:360])
                    r_tp8 = (c_tr[self.idx_tp8, self.idx_fc2] + c_tr[self.idx_tp8, self.idx_fc1]) / 2.0
                    r_tp7 = (c_tr[self.idx_tp7, self.idx_cp5] + c_tr[self.idx_f5, self.idx_fc5]) / 2.0
                    lat_lb_train.append(r_tp8 - r_tp7)

        feat_tr_list.append(flow_bb_tr)
        X_tr = np.concatenate(feat_tr_list, axis=1)
        print(f"Total Train Feature Shape: {X_tr.shape}")

        X_tr_s = self.scaler.fit_transform(X_tr)

        print("Fitting OvO Binary Specialists...")
        m_tr_01 = (y_all_tr == 0) | (y_all_tr == 1)
        self.ridge_01.fit(X_tr_s[m_tr_01], y_all_tr[m_tr_01])

        m_tr_02 = (y_all_tr == 0) | (y_all_tr == 2)
        self.ridge_02.fit(X_tr_s[m_tr_02], y_all_tr[m_tr_02])

        m_tr_12 = (y_all_tr == 1) | (y_all_tr == 2)
        self.ridge_12.fit(X_tr_s[m_tr_12], y_all_tr[m_tr_12])

        lat_lb_train = np.array(lat_lb_train)
        self.mu_lat = float(np.mean(lat_lb_train))
        self.s_lat = float(np.std(lat_lb_train) + 1e-12)

        self.fitted = True
        self.reset_state()
        print("Fitting complete cleanly.")

    def predict(self, X):
        X_np = to_numpy(X)
        if X_np.ndim == 2:
            X_np = X_np[np.newaxis, ...]

        if self.running_cov_e is None:
            self.reset_state()

        preds = []
        sos_bb = butter(4, [1.0, 40.0], btype="bandpass", fs=self.sfreq, output="sos")

        for trial_idx in range(len(X_np)):
            trial = X_np[trial_idx:trial_idx+1]
            self.n_test_windows += 1
            feat_curr = []

            X_bb = sosfiltfilt(sos_bb, trial, axis=-1)
            sf_bb = np.mean(X_bb[0, self.f_idx], axis=0)
            sp_bb = np.mean(X_bb[0, self.p_idx], axis=0)
            sf_bb = (sf_bb - sf_bb.mean()) / (sf_bb.std() + 1e-12)
            sp_bb = (sp_bb - sp_bb.mean()) / (sp_bb.std() + 1e-12)
            flow_bb_val = np.array([[np.mean(sf_bb[:-self.tau] * sp_bb[self.tau:]) - np.mean(sp_bb[:-self.tau] * sf_bb[self.tau:])]])

            lat_val = 0.0

            for b, (fl, fh) in enumerate(self.bands):
                X_filt = self._filter_band(trial, fl, fh)
                w_e = X_filt[..., :240]
                w_l = X_filt[..., 240:]

                c_e = self.cov_early[b].transform(w_e)[0]
                c_l = self.cov_late[b].transform(w_l)[0]
                self.running_cov_e[b] += c_e
                self.running_cov_l[b] += c_l

                ref_e_test = self.running_cov_e[b] / self.n_test_windows
                ref_l_test = self.running_cov_l[b] / self.n_test_windows

                inv_e = _invsqrtm(ref_e_test)
                inv_l = _invsqrtm(ref_l_test)

                c_e_al = (inv_e @ c_e @ inv_e)[np.newaxis, ...]
                c_l_al = (inv_l @ c_l @ inv_l)[np.newaxis, ...]

                corr_e, diag_e = _cov_to_corr(c_e_al)
                corr_l, diag_l = _cov_to_corr(c_l_al)

                conn_e = self.ts_early[b].transform(corr_e)
                conn_l = self.ts_late[b].transform(corr_l)

                p_e = np.log(np.maximum(diag_e, 1e-12))
                p_l = np.log(np.maximum(diag_l, 1e-12))
                p_macro = p_l - p_e

                p_stages = []
                for s, (t0, t1) in enumerate(self.stages):
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

                pF_stages = np.mean(p_norm[:, :, self.f_idx], axis=-1)
                pM_stages = np.mean(p_norm[:, :, self.m_idx], axis=-1)
                pP_stages = np.mean(p_norm[:, :, self.p_idx], axis=-1)

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

                sf_b = np.mean(X_filt[0, self.f_idx], axis=0)
                sp_b = np.mean(X_filt[0, self.p_idx], axis=0)
                sf_b = (sf_b - sf_b.mean()) / (sf_b.std() + 1e-12)
                sp_b = (sp_b - sp_b.mean()) / (sp_b.std() + 1e-12)
                flow_b_val = np.array([[np.mean(sf_b[:-self.tau] * sp_b[self.tau:]) - np.mean(sp_b[:-self.tau] * sf_b[self.tau:])]])

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

                if b == 3:
                    c_tr = np.corrcoef(X_filt[0, :, 120:360])
                    r_tp8 = (c_tr[self.idx_tp8, self.idx_fc2] + c_tr[self.idx_tp8, self.idx_fc1]) / 2.0
                    r_tp7 = (c_tr[self.idx_tp7, self.idx_cp5] + c_tr[self.idx_f5, self.idx_fc5]) / 2.0
                    lat_val = float(r_tp8 - r_tp7)

            feat_curr.append(flow_bb_val)
            X_vec = np.concatenate(feat_curr, axis=1)
            X_sc = self.scaler.transform(X_vec)

            # Solution A: Direct raw margins (no arbitrary scaling denominators!)
            raw01 = float(self.ridge_01.decision_function(X_sc)[0])
            raw02 = float(self.ridge_02.decision_function(X_sc)[0])
            raw12 = float(self.ridge_12.decision_function(X_sc)[0])

            ms_raw = -(raw01 + raw02) / 2.0

            # Causal Bayesian lateralization centering
            th_lat = (self.r_lat + self.M_lat * self.mu_lat) / (self.M_lat + self.n_lat)
            z_lat = (lat_val - th_lat) / self.s_lat
            self.r_lat += lat_val
            self.n_lat += 1

            th_mot = (self.r_mot / (self.M_mot + self.n_mot)) + self.base_th_raw
            th_cog = (self.r_cog / (self.M_cog + self.n_cog))

            self.r_mot += ms_raw
            self.n_mot += 1

            eff_12 = (raw12 - th_cog) - (self.w_lat_raw * 0.16) * z_lat

            # Solution A: Cognitive Confidence Guard on raw margin scale (guard_raw = 0.60)
            if ms_raw > th_mot and abs(eff_12) < self.guard_raw:
                pred = 0
            else:
                pred = 2 if eff_12 > 0 else 1
                self.r_cog += raw12
                self.n_cog += 1

            if getattr(self, "record_trace", False):
                if not hasattr(AdaptiveGuardedHierarchicalModel, "global_traces"):
                    AdaptiveGuardedHierarchicalModel.global_traces = []
                AdaptiveGuardedHierarchicalModel.global_traces.append({
                    "raw01": raw01,
                    "raw02": raw02,
                    "raw12": raw12,
                    "ms_raw": ms_raw,
                    "th_mot": th_mot,
                    "th_cog": th_cog,
                    "eff_12": eff_12,
                    "lat_val": lat_val,
                    "z_lat": z_lat,
                    "pred": pred,
                    "gate_open": bool(ms_raw > th_mot),
                    "guard_triggered": bool(ms_raw > th_mot and abs(eff_12) >= self.guard_raw),
                })

            preds.append(pred)

        return np.array(preds, dtype=np.int64)


class Solver(CompetSolver):
    """Official competition solver for Adaptive Guarded Hierarchical Model."""

    name = "AdaptiveGuardedHierarchical"
    requirements = ["pip::pyriemann"]

    def load_model(self, meta):
        device = meta.get("device", getattr(self, "device", "cpu"))
        sfreq = meta.get("sfreq", 120.0)
        model = AdaptiveGuardedHierarchicalModel(
            n_chans=meta["n_chans"],
            n_classes=meta["n_classes"],
            sfreq=sfreq,
            device=device,
        )

        weights_joblib = meta["submission_dir"] / "weights.joblib"
        if weights_joblib.exists() and getattr(self, "train_loader", None) is None:
            print(f"[loading] {weights_joblib} into {self.name}")
            artifact = joblib.load(weights_joblib)
            model.cov_early = artifact["cov_early"]
            model.cov_late = artifact["cov_late"]
            model.ts_early = artifact["ts_early"]
            model.ts_late = artifact["ts_late"]
            model.ref_e_train = artifact["ref_e_train"]
            model.ref_l_train = artifact["ref_l_train"]
            model.scaler = artifact["scaler"]
            model.ridge_01 = artifact["ridge_01"]
            model.ridge_02 = artifact["ridge_02"]
            model.ridge_12 = artifact["ridge_12"]
            model.mu_lat = artifact["mu_lat"]
            model.s_lat = artifact["s_lat"]
            model.base_th_raw = artifact.get("base_th_raw", 0.01)
            model.guard_raw = artifact.get("guard_raw", 0.60)
            model.w_lat_raw = artifact.get("w_lat_raw", 2.0)
            model.M_lat = artifact.get("M_lat", 15)
            model.prior_N0 = artifact.get("prior_N0", 25)
            model.fitted = True

        return model

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        artifact = {
            "cov_early": model.cov_early,
            "cov_late": model.cov_late,
            "ts_early": model.ts_early,
            "ts_late": model.ts_late,
            "ref_e_train": model.ref_e_train,
            "ref_l_train": model.ref_l_train,
            "scaler": model.scaler,
            "ridge_01": model.ridge_01,
            "ridge_02": model.ridge_02,
            "ridge_12": model.ridge_12,
            "mu_lat": model.mu_lat,
            "s_lat": model.s_lat,
            "base_th_raw": model.base_th_raw,
            "guard_raw": model.guard_raw,
            "w_lat_raw": model.w_lat_raw,
            "M_lat": model.M_lat,
            "prior_N0": model.prior_N0,
            "sfreq": model.sfreq,
            "n_chans": model.n_chans,
            "n_classes": model.n_classes,
        }
        joblib.dump(artifact, path / "weights.joblib")
