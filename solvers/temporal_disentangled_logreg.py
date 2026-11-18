"""Temporal Disentangled Logistic Regression (TDLR) solver for Track 02 BCI Decoding.

Step 2 of First-Principles Optimization:
Evaluates true Cross-Entropy loss (Logistic Regression) vs squared error (Ridge).

Hypothesis:
- Ridge penalizes 'overly confident correct' predictions (+3.5) because it minimizes
  squared error on continuous targets.
- Logistic Regression optimizes Cross-Entropy loss directly on class probabilities.
  It rewards confident correct classifications and outputs calibrated probability distributions.

Feature Backbone:
- Identical to the 44.80% TemporalDisentangledRiemann:
  - 6 sub-bands: Theta (4-8), Low-Mu (8-10.5), High-Mu (10.5-13),
    Low-Beta (13-20), High-Beta (20-28), Low-Gamma (28-42).
  - Early (1-3s) and Late (3-5s) epochs sliced independently to preserve the 3.0s boundary.
  - Scale-invariant Ledoit-Wolf correlation tangent features (10,332).
  - Local electrode log-power and explicit temporal delta (738).
  - Total: 11,070 features.
"""

import joblib
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


class TemporalDisentangledLogRegModel:
    """Disentangled Riemannian Pipeline with Cross-Entropy Logistic Regression."""

    def __init__(self, sfreq=120.0, bands=None, C=0.05, max_iter=500):
        self.sfreq = float(sfreq)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.bands = bands or [
            (4.0, 8.0),    # Theta (4-8 Hz) - Frontal cognitive workload
            (8.0, 10.5),   # Low Mu/Alpha (8-10.5 Hz) - SMR idling / alpha floor
            (10.5, 13.0),  # High Mu/Alpha (10.5-13 Hz) - SMR ERD (Motor Imagery)
            (13.0, 20.0),  # Low Beta (13-20 Hz) - Motor execution & activation
            (20.0, 28.0),  # High Beta (20-28 Hz) - Motor planning & control
            (28.0, 42.0),  # Low Gamma (28-42 Hz) - High-level verbal cognition
        ]
        self.n_bands = len(self.bands)

        # Early epoch estimators
        self.cov_early = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_early = [TangentSpace(metric="riemann") for _ in self.bands]

        # Late epoch estimators
        self.cov_late = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_late = [TangentSpace(metric="riemann") for _ in self.bands]

        self.scaler = StandardScaler()
        # L2-regularized Logistic Regression optimizing cross-entropy loss
        self.clf = LogisticRegression(
            C=self.C, penalty="l2", solver="lbfgs", max_iter=self.max_iter
        )
        self.fitted = False

    def _filter_band(self, X, f_low, f_high):
        """Zero-phase Butterworth bandpass filter along the time axis."""
        sos = butter(4, [f_low, f_high], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)

    def _cov_to_corr(self, covs):
        """Convert strictly positive-definite shrinkage covariances to correlation matrices."""
        diags = np.diagonal(covs, axis1=-2, axis2=-1)
        stds = np.sqrt(np.maximum(diags, 1e-12))[..., np.newaxis]
        corrs = covs / (stds @ stds.swapaxes(-2, -1))
        for i in range(len(corrs)):
            np.fill_diagonal(corrs[i], 1.0)
        return corrs, diags

    def _extract_disentangled_epoch(self, X_epoch, cov_estimators, ts_estimators, fit=False):
        """Extract explicit log-power (41 per band) and correlation tangent features (861 per band)."""
        all_power = []
        all_conn = []

        for i, (f_low, f_high) in enumerate(self.bands):
            X_b = self._filter_band(X_epoch, f_low, f_high)
            if fit:
                covs = cov_estimators[i].fit_transform(X_b)
                corrs, diags = self._cov_to_corr(covs)
                conn_feats = ts_estimators[i].fit_transform(corrs)
            else:
                covs = cov_estimators[i].transform(X_b)
                corrs, diags = self._cov_to_corr(covs)
                conn_feats = ts_estimators[i].transform(corrs)

            # Local power: log-variance across channels
            log_power = np.log(np.maximum(diags, 1e-12))
            all_power.append(log_power)
            all_conn.append(conn_feats)

        power_feats = np.concatenate(all_power, axis=1)  # (B, 6 * 41 = 246)
        conn_feats = np.concatenate(all_conn, axis=1)    # (B, 6 * 861 = 5166)
        return power_feats, conn_feats

    def fit(self, train_loader):
        # 1. Gather all training trials
        X_all, y_all = [], []
        for X, y, _info in train_loader:
            X_all.append(to_numpy(X))
            y_np = to_numpy(y)
            if y_np.ndim > 1 and y_np.shape[-1] == 3:
                y_all.append(y_np.argmax(axis=-1))
            else:
                y_all.append(y_np)

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        # 2. Slice into Early Phase (0-240) and Late Phase (240-480)
        split_idx = X_all.shape[-1] // 2
        X_early = X_all[..., :split_idx]
        X_late = X_all[..., split_idx:]

        # 3. Extract Disentangled Features for both epochs independently
        power_early, conn_early = self._extract_disentangled_epoch(
            X_early, self.cov_early, self.ts_early, fit=True
        )
        power_late, conn_late = self._extract_disentangled_epoch(
            X_late, self.cov_late, self.ts_late, fit=True
        )

        # 4. Compute Explicit Temporal Slope (Delta Power)
        power_delta = power_late - power_early

        # 5. Concatenate all feature streams: shape (N, 11070)
        feats_combined = np.concatenate(
            [power_early, power_late, power_delta, conn_early, conn_late], axis=1
        )
        feats_scaled = self.scaler.fit_transform(feats_combined)

        # 6. Fit Logistic Regression with Cross-Entropy
        self.clf.fit(feats_scaled, y_all)
        self.fitted = True
        return self

    def predict(self, X):
        X_np = to_numpy(X)
        if not self.fitted:
            return np.zeros(len(X_np), dtype=np.int64)

        if X_np.ndim == 2:
            X_np = X_np[np.newaxis, ...]

        split_idx = X_np.shape[-1] // 2
        X_early = X_np[..., :split_idx]
        X_late = X_np[..., split_idx:]

        power_early, conn_early = self._extract_disentangled_epoch(
            X_early, self.cov_early, self.ts_early, fit=False
        )
        power_late, conn_late = self._extract_disentangled_epoch(
            X_late, self.cov_late, self.ts_late, fit=False
        )

        power_delta = power_late - power_early

        feats_combined = np.concatenate(
            [power_early, power_late, power_delta, conn_early, conn_late], axis=1
        )
        feats_scaled = self.scaler.transform(feats_combined)
        return self.clf.predict(feats_scaled).astype(np.int64)


class Solver(CompetSolver):
    """Competition solver for Temporal Disentangled Logistic Regression."""

    name = "TemporalDisentangledLogReg"

    requirements = ["pip::pyriemann"]

    def load_model(self, meta):
        sfreq = meta.get("sfreq", 120.0)
        model = TemporalDisentangledLogRegModel(sfreq=sfreq)

        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists() and self.train_loader is None:
            print(f"[loading] {weights} into {self.name}")
            artifact = joblib.load(weights)
            model.cov_early = artifact["cov_early"]
            model.ts_early = artifact["ts_early"]
            model.cov_late = artifact["cov_late"]
            model.ts_late = artifact["ts_late"]
            model.scaler = artifact["scaler"]
            model.clf = artifact["clf"]
            model.sfreq = artifact["sfreq"]
            model.bands = artifact["bands"]
            model.fitted = True
        return model

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        artifact = {
            "cov_early": model.cov_early,
            "ts_early": model.ts_early,
            "cov_late": model.cov_late,
            "ts_late": model.ts_late,
            "scaler": model.scaler,
            "clf": model.clf,
            "sfreq": model.sfreq,
            "bands": model.bands,
        }
        joblib.dump(artifact, path / "weights.joblib")

