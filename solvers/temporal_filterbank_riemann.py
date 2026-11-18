"""Temporal Filter Bank Riemannian (TFBR) solver for Track 02 BCI Decoding.

Phase 1 of First-Principles Optimization: Temporal Dynamics.
Empirical basis:
- Analysis of 2,600+ PROTEUS trials shows that Mental Subtraction frontal power
  surges by +31% over the 4s window, while Motor Imagery drops by 15-19%.
- Single 4s covariance averages out this temporal trajectory.

Architecture:
1. Decomposes the 4-second window (480 samples) into two 2-second temporal epochs:
   - Early Epoch (samples 0-240): Open-loop mental command initiation.
   - Late Epoch (samples 240-480): Closed-loop feedback phase.
2. Extracts 6 neurophysiological sub-bands across both epochs:
   - Theta (4-8 Hz), Low-Mu (8-10.5 Hz), High-Mu (10.5-13 Hz),
     Low-Beta (13-20 Hz), High-Beta (20-28 Hz), Low-Gamma (28-42 Hz).
3. Independent Ledoit-Wolf shrinkage covariance and Riemannian Tangent Space
   modeling for early and late phases (5,166 features each -> 10,332 total).
4. Standardized L2-Regularized Ridge Classification on the full spatio-temporal fingerprint.
"""

import joblib
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


class TemporalFilterBankRiemannModel:
    """Two-Epoch Filter Bank Riemannian pipeline capturing temporal brain dynamics."""

    def __init__(self, sfreq=120.0, bands=None, alpha=150.0):
        self.sfreq = float(sfreq)
        self.alpha = float(alpha)
        self.bands = bands or [
            (4.0, 8.0),    # Theta (4-8 Hz) - Frontal cognitive workload
            (8.0, 10.5),   # Low Mu/Alpha (8-10.5 Hz) - SMR idling / alpha floor
            (10.5, 13.0),  # High Mu/Alpha (10.5-13 Hz) - SMR ERD (Motor Imagery)
            (13.0, 20.0),  # Low Beta (13-20 Hz) - Motor execution & activation
            (20.0, 28.0),  # High Beta (20-28 Hz) - Motor planning & control
            (28.0, 42.0),  # Low Gamma (28-42 Hz) - High-level verbal cognition
        ]
        self.n_bands = len(self.bands)

        # Separate Riemannian estimators for early (initiation) and late (feedback) phases
        self.cov_early = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_early = [TangentSpace(metric="riemann") for _ in self.bands]

        self.cov_late = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_late = [TangentSpace(metric="riemann") for _ in self.bands]

        self.scaler = StandardScaler()
        self.clf = RidgeClassifier(alpha=self.alpha)
        self.fitted = False

    def _filter_band(self, X, f_low, f_high):
        """Zero-phase Butterworth bandpass filter along the time axis."""
        sos = butter(4, [f_low, f_high], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)

    def _extract_epoch_features(self, X_epoch, cov_estimators, ts_estimators, fit=False):
        """Extract tangent space features for a given temporal epoch across all bands."""
        band_feats = []
        for i, (f_low, f_high) in enumerate(self.bands):
            X_b = self._filter_band(X_epoch, f_low, f_high)
            if fit:
                covs = cov_estimators[i].fit_transform(X_b)
                feats = ts_estimators[i].fit_transform(covs)
            else:
                covs = cov_estimators[i].transform(X_b)
                feats = ts_estimators[i].transform(covs)
            band_feats.append(feats)
        return np.concatenate(band_feats, axis=1)

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

        # 3. Fit Riemannian Tangent Space features for both epochs independently
        feats_early = self._extract_epoch_features(
            X_early, self.cov_early, self.ts_early, fit=True
        )
        feats_late = self._extract_epoch_features(
            X_late, self.cov_late, self.ts_late, fit=True
        )

        # 4. Concatenate early and late features: shape (N, 10332)
        feats_combined = np.concatenate([feats_early, feats_late], axis=1)
        feats_scaled = self.scaler.fit_transform(feats_combined)

        # 5. Fit regularized classifier
        self.clf.fit(feats_scaled, y_all)
        self.fitted = True
        return self

    def predict(self, X):
        X_np = to_numpy(X)
        if not self.fitted:
            return np.zeros(len(X_np), dtype=np.int64)

        if X_np.ndim == 2:
            X_np = X_np[np.newaxis, ...]

        # Slice into Early Phase (0-240) and Late Phase (240-480)
        split_idx = X_np.shape[-1] // 2
        X_early = X_np[..., :split_idx]
        X_late = X_np[..., split_idx:]

        feats_early = self._extract_epoch_features(
            X_early, self.cov_early, self.ts_early, fit=False
        )
        feats_late = self._extract_epoch_features(
            X_late, self.cov_late, self.ts_late, fit=False
        )

        feats_combined = np.concatenate([feats_early, feats_late], axis=1)
        feats_scaled = self.scaler.transform(feats_combined)
        return self.clf.predict(feats_scaled).astype(np.int64)


class Solver(CompetSolver):
    """Competition solver for Temporal Filter Bank Riemannian decoding."""

    name = "TemporalFilterBankRiemann"

    requirements = ["pip::pyriemann"]

    def load_model(self, meta):
        sfreq = meta.get("sfreq", 120.0)
        model = TemporalFilterBankRiemannModel(sfreq=sfreq)

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

