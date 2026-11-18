"""Unified 6-Band Riemann + Transformer Ensemble for Track 02 BCI Decoding.

Combines:
1. Shared 6-Sub-Band Riemannian Feature Backbone:
   - Decomposes each 4-second EEG window into 6 targeted neurophysiological bands:
     Theta (4-8 Hz), Low-Mu (8-10.5 Hz), High-Mu (10.5-13 Hz),
     Low-Beta (13-20 Hz), High-Beta (20-28 Hz), Low-Gamma (28-42 Hz).
   - Computes Ledoit-Wolf shrinkage covariances and Riemannian Tangent Space projections.
   - Extracts standardized tangent tokens of shape (B, 6, 861) and flattened (B, 5166).
2. Dual Complementary Predictors:
   - Regularized Linear Head: RidgeClassifier (alpha=100.0) on (B, 5166) features.
   - Deep Attention Head: CrossBandTransformerNet with dropout (0.50) on (B, 6, 861).
3. Soft-Probability Blending:
   - Converts decision margins and logits to probability distributions via Softmax.
   - P_ens = 0.55 * P_ridge + 0.45 * P_trans.
   - Takes the argmax class index.
"""

import copy
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


class CrossBandTransformerNet(nn.Module):
    """Compact Transformer learning cross-frequency interactions across 6 band tokens."""

    def __init__(
        self,
        n_bands=6,
        in_features=861,
        d_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_layers=1,
        dropout=0.50,
        n_classes=3,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.in_features = in_features
        self.d_model = d_model

        self.token_proj = nn.Linear(in_features, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.band_emb = nn.Parameter(torch.randn(1, n_bands, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        tokens = self.token_proj(x)
        tokens = self.norm_in(tokens) + self.band_emb
        tokens = self.transformer(tokens)
        tokens = self.norm_out(tokens)

        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.classifier(pooled)


class UnifiedEnsembleModel:
    """Unified 6-Band pipeline executing Ridge + Transformer on shared tangent representations."""

    def __init__(
        self,
        n_chans=41,
        n_classes=3,
        sfreq=120.0,
        device="cpu",
        alpha=100.0,
        d_model=64,
        n_heads=4,
        dropout=0.50,
        n_epochs=35,
        lr=1e-3,
        weight_decay=2e-3,
        label_smoothing=0.15,
        batch_size=32,
        weight_ridge=0.55,
    ):
        self.n_chans = n_chans
        self.n_classes = n_classes
        self.sfreq = float(sfreq)
        self.device = device
        self.weight_ridge = float(weight_ridge)

        self.bands = [
            (4.0, 8.0),    # Theta (4-8 Hz)
            (8.0, 10.5),   # Low Mu/Alpha (8-10.5 Hz)
            (10.5, 13.0),  # High Mu/Alpha (10.5-13 Hz)
            (13.0, 20.0),  # Low Beta (13-20 Hz)
            (20.0, 28.0),  # High Beta (20-28 Hz)
            (28.0, 42.0),  # Low Gamma (28-42 Hz)
        ]
        self.n_bands = len(self.bands)
        self.in_features = (n_chans * (n_chans + 1)) // 2

        # 1. Shared Feature Extraction Estimators
        self.cov_estimators = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_estimators = [TangentSpace(metric="riemann") for _ in self.bands]
        self.scalers = [StandardScaler() for _ in self.bands]

        # 2. Linear Ridge Predictor
        self.clf_ridge = RidgeClassifier(alpha=alpha)

        # 3. Deep Transformer Predictor
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size

        self.net = CrossBandTransformerNet(
            n_bands=self.n_bands,
            in_features=self.in_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=self.d_model * 2,
            n_layers=1,
            dropout=self.dropout,
            n_classes=self.n_classes,
        ).to(self.device)

        self.fitted = False

    def _filter_band(self, X, f_low, f_high):
        sos = butter(4, [f_low, f_high], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)

    def _extract_tangent_tokens(self, X_np, fit=False):
        """Extracts 6 standardized band tokens (B, 6, 861)."""
        tokens = []
        for i, (f_low, f_high) in enumerate(self.bands):
            X_b = self._filter_band(X_np, f_low, f_high)
            if fit:
                covs = self.cov_estimators[i].fit_transform(X_b)
                feats = self.ts_estimators[i].fit_transform(covs)
                feats_scaled = self.scalers[i].fit_transform(feats)
            else:
                covs = self.cov_estimators[i].transform(X_b)
                feats = self.ts_estimators[i].transform(covs)
                feats_scaled = self.scalers[i].transform(feats)
            tokens.append(feats_scaled)
        return np.stack(tokens, axis=1)

    def fit(self, train_loader):
        # 1. Gather all raw training windows
        X_all, y_all = [], []
        for X, y, _info in train_loader:
            X_all.append(to_numpy(X))
            y_np = to_numpy(y)
            if y_np.ndim > 1 and y_np.shape[-1] == self.n_classes:
                y_all.append(y_np.argmax(axis=-1))
            else:
                y_all.append(y_np)

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        # 2. Extract shared Riemannian tangent tokens: shape (N, 6, 861)
        tokens_all = self._extract_tangent_tokens(X_all, fit=True)

        # 3. Fit Linear Ridge Head on flattened features: shape (N, 5166)
        feats_flat = tokens_all.reshape(len(tokens_all), -1)
        self.clf_ridge.fit(feats_flat, y_all)

        # 4. Fit Deep Transformer Head on tokens: shape (N, 6, 861)
        dataset = TensorDataset(
            torch.tensor(tokens_all, dtype=torch.float32),
            torch.tensor(y_all, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs, eta_min=1e-5
        )
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        self.net.train()
        for epoch in range(self.n_epochs):
            for batch_tokens, batch_labels in loader:
                batch_tokens = batch_tokens.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.net(batch_tokens)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

        self.fitted = True
        return self

    @torch.no_grad()
    def predict(self, X):
        X_np = to_numpy(X)
        if not self.fitted:
            return np.zeros(len(X_np), dtype=np.int64)

        if X_np.ndim == 2:
            X_np = X_np[np.newaxis, ...]

        # 1. Extract shared 6-band tangent features once
        tokens = self._extract_tangent_tokens(X_np, fit=False)  # (B, 6, 861)
        feats_flat = tokens.reshape(len(tokens), -1)             # (B, 5166)

        # 2. Linear Ridge Probabilities (via Softmax on decision scores)
        margins = self.clf_ridge.decision_function(feats_flat)
        p_ridge = softmax(margins, axis=-1)

        # 3. Deep Transformer Probabilities (via Softmax on logits)
        tokens_tensor = torch.tensor(tokens, dtype=torch.float32, device=self.device)
        self.net.eval()
        logits = self.net(tokens_tensor)
        p_trans = torch.softmax(logits, dim=-1).cpu().numpy()

        # 4. Soft-Blended Prediction
        p_ens = (self.weight_ridge * p_ridge) + ((1.0 - self.weight_ridge) * p_trans)
        return p_ens.argmax(axis=-1).astype(np.int64)


class Solver(CompetSolver):
    """Competition solver for Unified Riemann + Transformer Ensemble."""

    name = "RiemannTransformerEnsemble"

    requirements = ["pip::pyriemann"]

    def load_model(self, meta):
        sfreq = meta.get("sfreq", 120.0)
        model = UnifiedEnsembleModel(
            n_chans=meta["n_chans"],
            n_classes=meta["n_classes"],
            sfreq=sfreq,
            device=self.device,
        )

        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists() and self.train_loader is None:
            print(f"[loading] {weights} into {self.name}")
            artifact = joblib.load(weights)

            model.cov_estimators = artifact["cov_estimators"]
            model.ts_estimators = artifact["ts_estimators"]
            model.scalers = artifact["scalers"]
            model.bands = artifact["bands"]
            model.sfreq = artifact["sfreq"]
            model.n_chans = artifact["n_chans"]
            model.n_classes = artifact["n_classes"]
            model.n_bands = len(model.bands)
            model.weight_ridge = artifact.get("weight_ridge", 0.55)

            model.clf_ridge = artifact["clf_ridge"]

            model.net = CrossBandTransformerNet(
                n_bands=model.n_bands,
                in_features=model.in_features,
                d_model=model.d_model,
                n_heads=model.n_heads,
                dim_feedforward=model.d_model * 2,
                n_layers=1,
                dropout=model.dropout,
                n_classes=model.n_classes,
            ).to(self.device)
            model.net.load_state_dict(artifact["net_state_dict"])
            model.net.to(self.device)
            model.fitted = True

        return model

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        artifact = {
            "cov_estimators": model.cov_estimators,
            "ts_estimators": model.ts_estimators,
            "scalers": model.scalers,
            "bands": model.bands,
            "sfreq": model.sfreq,
            "n_chans": model.n_chans,
            "n_classes": model.n_classes,
            "clf_ridge": model.clf_ridge,
            "net_state_dict": {k: v.cpu() for k, v in model.net.state_dict().items()},
            "weight_ridge": model.weight_ridge,
        }
        joblib.dump(artifact, path / "weights.joblib")
        torch.save(model.net.state_dict(), path / "weights.pt")
