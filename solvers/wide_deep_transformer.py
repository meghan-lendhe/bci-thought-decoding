"""Wide-and-Deep Temporal Transformer (WDTT) solver for Track 02 BCI Decoding.

Phase 4 of First-Principles Optimization:
1. Feature Backbone (Identical to 44.80% TemporalDisentangledRiemann):
   - 6 Sub-Bands: Theta (4-8 Hz), Low-Mu (8-10.5 Hz), High-Mu (10.5-13 Hz),
     Low-Beta (13-20 Hz), High-Beta (20-28 Hz), Low-Gamma (28-42 Hz).
   - Early (1-3s) and Late (3-5s) temporal epochs.
   - Scale-invariant Ledoit-Wolf correlation tangent features (861 per band per epoch).
   - Local electrode log-power (41 per band per epoch) and explicit temporal delta.
   - Total of 1,845 features per band -> 11,070 total features.
2. Wide-and-Deep Architecture:
   - WIDE Path: Direct linear highway on all 11,070 standardized features,
     anchoring the network with the full power of the 44.80% linear baseline.
   - DEEP Path: Widen token dimension to d=128 (no extreme bottleneck).
     Pre-LN Multi-Head Self-Attention across the 6 band tokens (each with 1,845 features)
     with dropout=0.50 and dual pooling (mean + max).
   - Fusion: Logits = Wide_Logits + Deep_Residual_Logits.
3. Optimization:
   - AdamW with weight decay (3e-3), Cosine Annealing (35 epochs), and Label Smoothing (0.15).
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
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


class WideDeepTransformerNet(nn.Module):
    """Wide-and-Deep Architecture combining direct linear highway and cross-band attention."""

    def __init__(
        self,
        n_bands=6,
        in_features_per_band=1845,
        d_model=128,
        n_heads=4,
        dim_feedforward=256,
        n_layers=1,
        dropout=0.50,
        n_classes=3,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.in_features_per_band = in_features_per_band
        self.total_features = n_bands * in_features_per_band
        self.d_model = d_model

        # 1. WIDE PATH: Direct linear highway on all raw features
        self.wide_linear = nn.Linear(self.total_features, n_classes)

        # 2. DEEP PATH: Token projection and Cross-Band Transformer
        self.token_proj = nn.Linear(in_features_per_band, d_model)
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

        self.deep_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x_flat, x_tokens):
        # 1. Wide prediction (anchors the baseline)
        wide_logits = self.wide_linear(x_flat)

        # 2. Deep attention prediction (learns non-linear band interactions)
        tokens = self.token_proj(x_tokens)
        tokens = self.norm_in(tokens) + self.band_emb
        tokens = self.transformer(tokens)
        tokens = self.norm_out(tokens)

        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        deep_logits = self.deep_head(pooled)

        # 3. Residual fusion
        return wide_logits + deep_logits


class WideDeepTransformerModel:
    """Complete pipeline: Disentangled Temporal Riemann + Wide-and-Deep Transformer."""

    def __init__(
        self,
        n_chans=41,
        n_classes=3,
        sfreq=120.0,
        bands=None,
        device="cpu",
        d_model=128,
        n_heads=4,
        n_layers=1,
        dropout=0.50,
        n_epochs=35,
        lr=1e-3,
        weight_decay=3e-3,
        label_smoothing=0.15,
        batch_size=32,
    ):
        self.n_chans = n_chans
        self.n_classes = n_classes
        self.sfreq = float(sfreq)
        self.bands = bands or [
            (4.0, 8.0),    # Theta (4-8 Hz)
            (8.0, 10.5),   # Low Mu/Alpha (8-10.5 Hz)
            (10.5, 13.0),  # High Mu/Alpha (10.5-13 Hz)
            (13.0, 20.0),  # Low Beta (13-20 Hz)
            (20.0, 28.0),  # High Beta (20-28 Hz)
            (28.0, 42.0),  # Low Gamma (28-42 Hz)
        ]
        self.device = device
        self.n_bands = len(self.bands)

        # Tangent dimension = C*(C+1)//2 = 861
        self.tangent_dim = (n_chans * (n_chans + 1)) // 2
        # Per-band features: power_early(41) + power_late(41) + power_delta(41) + conn_early(861) + conn_late(861)
        self.in_features_per_band = (n_chans * 3) + (self.tangent_dim * 2)
        self.total_features = self.n_bands * self.in_features_per_band

        # Estimators
        self.cov_early = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_early = [TangentSpace(metric="riemann") for _ in self.bands]
        self.cov_late = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_late = [TangentSpace(metric="riemann") for _ in self.bands]

        self.scaler = StandardScaler()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size

        self.net = WideDeepTransformerNet(
            n_bands=self.n_bands,
            in_features_per_band=self.in_features_per_band,
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=self.d_model * 2,
            n_layers=self.n_layers,
            dropout=self.dropout,
            n_classes=self.n_classes,
        ).to(self.device)

        self.fitted = False

    def _filter_band(self, X, f_low, f_high):
        sos = butter(4, [f_low, f_high], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)

    def _cov_to_corr(self, covs):
        diags = np.diagonal(covs, axis1=-2, axis2=-1)
        stds = np.sqrt(np.maximum(diags, 1e-12))[..., np.newaxis]
        corrs = covs / (stds @ stds.swapaxes(-2, -1))
        for i in range(len(corrs)):
            np.fill_diagonal(corrs[i], 1.0)
        return corrs, diags

    def _extract_disentangled_bands(self, X_all, fit=False):
        """Extracts structured per-band features and returns (B, n_bands, in_features_per_band)."""
        split_idx = X_all.shape[-1] // 2
        X_early = X_all[..., :split_idx]
        X_late = X_all[..., split_idx:]

        band_tokens = []
        for i, (f_low, f_high) in enumerate(self.bands):
            X_e_b = self._filter_band(X_early, f_low, f_high)
            X_l_b = self._filter_band(X_late, f_low, f_high)

            if fit:
                cov_e = self.cov_early[i].fit_transform(X_e_b)
                corr_e, diag_e = self._cov_to_corr(cov_e)
                conn_e = self.ts_early[i].fit_transform(corr_e)

                cov_l = self.cov_late[i].fit_transform(X_l_b)
                corr_l, diag_l = self._cov_to_corr(cov_l)
                conn_l = self.ts_late[i].fit_transform(corr_l)
            else:
                cov_e = self.cov_early[i].transform(X_e_b)
                corr_e, diag_e = self._cov_to_corr(cov_e)
                conn_e = self.ts_early[i].transform(corr_e)

                cov_l = self.cov_late[i].transform(X_l_b)
                corr_l, diag_l = self._cov_to_corr(cov_l)
                conn_l = self.ts_late[i].transform(corr_l)

            p_early = np.log(np.maximum(diag_e, 1e-12))
            p_late = np.log(np.maximum(diag_l, 1e-12))
            p_delta = p_late - p_early

            # Per-band vector: [p_early (41), p_late (41), p_delta (41), conn_e (861), conn_l (861)] = 1845
            band_vec = np.concatenate([p_early, p_late, p_delta, conn_e, conn_l], axis=1)
            band_tokens.append(band_vec)

        # Stack into (B, n_bands, in_features_per_band)
        tokens_3d = np.stack(band_tokens, axis=1)
        B, n_b, n_f = tokens_3d.shape
        flat_2d = tokens_3d.reshape(B, n_b * n_f)

        if fit:
            flat_scaled = self.scaler.fit_transform(flat_2d)
        else:
            flat_scaled = self.scaler.transform(flat_2d)

        tokens_scaled = flat_scaled.reshape(B, n_b, n_f)
        return flat_scaled, tokens_scaled

    def fit(self, train_loader):
        # 1. Gather all training trials
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

        # 2. Extract features
        flat_all, tokens_all = self._extract_disentangled_bands(X_all, fit=True)

        # 3. Train Wide-and-Deep Network
        dataset = TensorDataset(
            torch.tensor(flat_all, dtype=torch.float32),
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
            for b_flat, b_tokens, b_labels in loader:
                b_flat = b_flat.to(self.device)
                b_tokens = b_tokens.to(self.device)
                b_labels = b_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.net(b_flat, b_tokens)
                loss = loss_fn(logits, b_labels)
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

        flat, tokens = self._extract_disentangled_bands(X_np, fit=False)
        flat_tensor = torch.tensor(flat, dtype=torch.float32, device=self.device)
        tokens_tensor = torch.tensor(tokens, dtype=torch.float32, device=self.device)

        self.net.eval()
        logits = self.net(flat_tensor, tokens_tensor)
        preds = logits.argmax(dim=-1).cpu().numpy()
        return preds.astype(np.int64)


class Solver(CompetSolver):
    """Competition solver for Wide-and-Deep Temporal Transformer."""

    name = "WideDeepTransformer"

    requirements = ["pip::pyriemann"]

    def load_model(self, meta):
        sfreq = meta.get("sfreq", 120.0)
        model = WideDeepTransformerModel(
            n_chans=meta["n_chans"],
            n_classes=meta["n_classes"],
            sfreq=sfreq,
            device=self.device,
        )

        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists() and self.train_loader is None:
            print(f"[loading] {weights} into {self.name}")
            artifact = joblib.load(weights)
            model.cov_early = artifact["cov_early"]
            model.ts_early = artifact["ts_early"]
            model.cov_late = artifact["cov_late"]
            model.ts_late = artifact["ts_late"]
            model.scaler = artifact["scaler"]
            model.bands = artifact["bands"]
            model.sfreq = artifact["sfreq"]
            model.n_chans = artifact["n_chans"]
            model.n_classes = artifact["n_classes"]
            model.n_bands = len(model.bands)

            model.net = WideDeepTransformerNet(
                n_bands=model.n_bands,
                in_features_per_band=model.in_features_per_band,
                d_model=model.d_model,
                n_heads=model.n_heads,
                dim_feedforward=model.d_model * 2,
                n_layers=model.n_layers,
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
            "cov_early": model.cov_early,
            "ts_early": model.ts_early,
            "cov_late": model.cov_late,
            "ts_late": model.ts_late,
            "scaler": model.scaler,
            "bands": model.bands,
            "sfreq": model.sfreq,
            "n_chans": model.n_chans,
            "n_classes": model.n_classes,
            "net_state_dict": {k: v.cpu() for k, v in model.net.state_dict().items()},
        }
        joblib.dump(artifact, path / "weights.joblib")
        torch.save(model.net.state_dict(), path / "weights.pt")

