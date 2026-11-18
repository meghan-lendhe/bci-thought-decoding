"""Filter Bank Riemannian (FBR) solver for Track 02 BCI Decoding.

Decomposes raw EEG windows into four neurophysiological frequency bands:
  - Band 1: Theta (4-8 Hz) - Frontal cognitive workload (Mental Subtraction)
  - Band 2: Alpha/Mu (8-14 Hz) - Sensorimotor rhythm ERD/ERS (Motor Imagery)
  - Band 3: Beta (14-28 Hz) - Motor planning and execution
  - Band 4: Low Gamma (28-42 Hz) - High-level verbal cognition (Word Generation)

For each band, it computes Ledoit-Wolf shrinkage covariance matrices, projects
them onto the Riemannian Tangent Space manifold, concatenates the resulting
spatial features, and classifies using a regularized Ridge classifier.
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


class FilterBankRiemannModel:
    """Filter Bank Riemannian Pipeline with Ridge classification."""

    def __init__(self, sfreq=120.0, bands=None, alpha=100.0):
        self.sfreq = float(sfreq)
        self.bands = bands or [
            (4.0, 8.0),    # Band 1: Theta (4-8 Hz) - Frontal cognitive workload
            (8.0, 10.5),   # Band 2: Low Alpha/Mu (8-10.5 Hz) - SMR idling / alpha floor
            (10.5, 13.0),  # Band 3: High Alpha/Mu (10.5-13 Hz) - SMR ERD (Motor Imagery)
            (13.0, 20.0),  # Band 4: Low Beta (13-20 Hz) - Motor execution & activation
            (20.0, 28.0),  # Band 5: High Beta (20-28 Hz) - Motor planning & control
            (28.0, 42.0),  # Band 6: Low Gamma (28-42 Hz) - High-level verbal cognition
        ]
        self.alpha = alpha
        self.cov_estimators = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_estimators = [TangentSpace(metric="riemann") for _ in self.bands]
        self.scaler = StandardScaler()
        self.clf = RidgeClassifier(alpha=self.alpha)
        self.fitted = False

    def _filter_band(self, X, f_low, f_high):
        """Zero-phase bandpass filter along the time axis."""
        sos = butter(4, [f_low, f_high], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)

    def fit(self, train_loader):
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

        # Feature extraction across bands
        all_band_feats = []
        for i, (f_low, f_high) in enumerate(self.bands):
            X_b = self._filter_band(X_all, f_low, f_high)
            covs = self.cov_estimators[i].fit_transform(X_b)
            feats = self.ts_estimators[i].fit_transform(covs)
            all_band_feats.append(feats)

        feats_combined = np.concatenate(all_band_feats, axis=1)
        feats_scaled = self.scaler.fit_transform(feats_combined)
        self.clf.fit(feats_scaled, y_all)
        self.fitted = True
        return self

    def predict(self, X):
        X_np = to_numpy(X)
        if not self.fitted:
            return np.zeros(len(X_np), dtype=np.int64)

        # Ensure 3D shape (B, C, T) even for single trial
        if X_np.ndim == 2:
            X_np = X_np[np.newaxis, ...]

        all_band_feats = []
        for i, (f_low, f_high) in enumerate(self.bands):
            X_b = self._filter_band(X_np, f_low, f_high)
            covs = self.cov_estimators[i].transform(X_b)
            feats = self.ts_estimators[i].transform(covs)
            all_band_feats.append(feats)

        feats_combined = np.concatenate(all_band_feats, axis=1)
        feats_scaled = self.scaler.transform(feats_combined)
        return self.clf.predict(feats_scaled).astype(np.int64)


class Solver(CompetSolver):
    """Competition solver wrapper for Filter Bank Riemannian decoding."""

    name = "FilterBankRiemann"

    requirements = ["pip::pyriemann"]

    def load_model(self, meta):
        sfreq = meta.get("sfreq", 120.0)
        model = FilterBankRiemannModel(sfreq=sfreq)
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists() and self.train_loader is None:
            print(f"[loading] {weights} into {self.name}")
            artifact = joblib.load(weights)
            model.cov_estimators = artifact["cov_estimators"]
            model.ts_estimators = artifact["ts_estimators"]
            model.scaler = artifact["scaler"]
            model.clf = artifact["clf"]
            model.sfreq = artifact["sfreq"]
            model.bands = artifact["bands"]
            model.fitted = True
        return model

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        # Save only standard pyriemann/sklearn estimators in a dict
        artifact = {
            "cov_estimators": model.cov_estimators,
            "ts_estimators": model.ts_estimators,
            "scaler": model.scaler,
            "clf": model.clf,
            "sfreq": model.sfreq,
            "bands": model.bands,
        }
        joblib.dump(artifact, path / "weights.joblib")
