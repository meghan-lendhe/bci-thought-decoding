"""Spatio-Temporal Deep Transformer (STDT) solver for Track 02 BCI Decoding.

Phase 4 Advanced: 12-Token Spatio-Temporal Disentangled Architecture:
1. Feature Extraction:
   - Slices the 480-sample window into Early (0-240, 1-3s) and Late (240-480, 3-5s) epochs
     BEFORE filtering, strictly preserving the 3.0s start_feedback boundary without
     acausal backward smearing.
   - 6 Sub-Bands: Theta (4-8 Hz), Low-Mu (8-10.5 Hz), High-Mu (10.5-13 Hz),
     Low-Beta (13-20 Hz), High-Beta (20-28 Hz), Low-Gamma (28-42 Hz).
   - For each band and epoch:
     * Ledoit-Wolf shrinkage covariance normalized to [-1, 1] correlation manifold
     * Local electrode log-power (41 features)
     * Riemannian tangent space connectivity (861 features)
     * Combined into 1 spatio-temporal token (902 features)
   - Total: 6 bands x 2 epochs = 12 Spatio-Temporal Tokens (10,824 features).
   - Plus explicit temporal slope (Delta Power = Power_Late - Power_Early, 246 features).
   - Total Flat Features: 11,070 standardized features.

2. Wide-and-Deep Residual Network:
   - WIDE Path: Direct linear highway on all 11,070 features (anchored at our 44.80% baseline).
   - DEEP Path:
     * Projects 12 tokens from 902 -> d=128
     * Factorized positional embeddings (band_emb[6] + time_emb[2])
     * Pre-LN Multi-Head Self-Attention (4 heads, dropout=0.50) attending across
       both frequency bands and temporal phases
     * Dual pooling (mean + max) -> 256-dim representation
     * Deep MLP head -> 3 class logits
   - Residual Fusion: Logits = Wide_Logits + gamma * Deep_Logits (gamma initialized to 0.20).

3. Optimization:
   - AdamW (lr=1e-3, weight_decay=3e-3), CosineAnnealing (35 epochs), Label Smoothing (0.15).
"""

import copy
import joblib
import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


class SpatioTemporalTransformerNet(nn.Module):
    """Wide-and-Deep Spatio-Temporal Architecture.
    
    Combines direct linear highway on all 11,070 features with a 12-token
    spatio-temporal self-attention mechanism over frequency and time.
    """

    def __init__(
        self,
        n_bands=6,
        n_epochs=2,
        in_features_per_token=902,
        n_delta_features=246,
        d_model=128,
        n_heads=4,
        dim_feedforward=256,
        n_layers=1,
        dropout=0.50,
        n_classes=3,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_epochs = n_epochs
        self.n_tokens = n_bands * n_epochs  # 12 tokens
        self.in_features_per_token = in_features_per_token
        self.total_token_features = self.n_tokens * in_features_per_token  # 10,824
        self.total_flat_features = self.total_token_features + n_delta_features  # 11,070
        self.d_model = d_model

        # 1. WIDE PATH: Direct linear highway on all 11,070 features
        self.wide_linear = nn.Linear(self.total_flat_features, n_classes)

        # 2. DEEP PATH: Token projection and Spatio-Temporal Attention
        self.token_proj = nn.Linear(in_features_per_token, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        # Factorized Positional Embeddings: band (6) and time (2)
        self.band_emb = nn.Parameter(torch.randn(1, n_bands, 1, d_model) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(1, 1, n_epochs, d_model) * 0.02)

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

        # Dual pooling (mean + max = d_model * 2) -> Deep classification head
        self.deep_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        # Residual gating scalar (anchored near linear highway initially)
        self.gamma = nn.Parameter(torch.tensor(0.20))

    def forward(self, x_flat, x_tokens):
        # 1. Wide prediction (anchors baseline)
        wide_logits = self.wide_linear(x_flat)

        # 2. Deep spatio-temporal attention prediction
        tokens = self.token_proj(x_tokens)
        tokens = self.norm_in(tokens)

        # Broadcast and reshape factorized embeddings across 12 tokens:
        # (1, 6, 2, d_model) -> (1, 12, d_model)
        pos_emb = (self.band_emb + self.time_emb).reshape(1, self.n_tokens, self.d_model)
        tokens = tokens + pos_emb

        # Cross-token multi-head attention (frequency + time)
        tokens = self.transformer(tokens)
        tokens = self.norm_out(tokens)

        # Dual pooling
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        deep_logits = self.deep_head(pooled)

        # 3. Residual fusion
        return wide_logits + self.gamma * deep_logits


class SpatioTemporalTransformerModel:
    """Disentangled 12-Token Pipeline with Wide Linear Highway and Spatio-Temporal Attention."""

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
        n_epochs_train=35,
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
        self.n_time_epochs = 2  # Early and Late
        self.n_tokens = self.n_bands * self.n_time_epochs  # 12

        # Tangent dimension = C*(C+1)//2 = 861
        self.tangent_dim = (n_chans * (n_chans + 1)) // 2
        # Token dimension = log_power (41) + tangent (861) = 902
        self.in_features_per_token = self.n_chans + self.tangent_dim
        # Delta features = 6 bands * 41 = 246
        self.n_delta_features = self.n_bands * self.n_chans
        # Total flat features = 12 * 902 + 246 = 11,070
        self.total_flat_features = (self.n_tokens * self.in_features_per_token) + self.n_delta_features

        # Estimators for Early (0-240) and Late (240-480)
        self.cov_early = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_early = [TangentSpace(metric="riemann") for _ in self.bands]
        self.cov_late = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_late = [TangentSpace(metric="riemann") for _ in self.bands]

        self.scaler = StandardScaler()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_epochs_train = n_epochs_train
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size

        self.net = SpatioTemporalTransformerNet(
            n_bands=self.n_bands,
            n_epochs=self.n_time_epochs,
            in_features_per_token=self.in_features_per_token,
            n_delta_features=self.n_delta_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=self.d_model * 2,
            n_layers=self.n_layers,
            dropout=self.dropout,
            n_classes=self.n_classes,
        ).to(self.device)

        self.fitted = False

    def _filter_band(self, X, f_low, f_high):
        """Zero-phase Butterworth bandpass filter along time axis."""
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

    def _extract_disentangled_tokens(self, X_all, fit=False):
        """Extracts 12 Spatio-Temporal tokens and flat feature vector."""
        split_idx = X_all.shape[-1] // 2
        X_early = X_all[..., :split_idx]
        X_late = X_all[..., split_idx:]

        tokens_list = []
        delta_list = []

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

            p_early = np.log(np.maximum(diag_e, 1e-12))  # (B, 41)
            p_late = np.log(np.maximum(diag_l, 1e-12))   # (B, 41)
            p_delta = p_late - p_early                   # (B, 41)

            # Token 2*i: Early (log_power + tangent) = 902 features
            token_early = np.concatenate([p_early, conn_e], axis=1)
            # Token 2*i + 1: Late (log_power + tangent) = 902 features
            token_late = np.concatenate([p_late, conn_l], axis=1)

            tokens_list.append(token_early)
            tokens_list.append(token_late)
            delta_list.append(p_delta)

        # Stack into (B, 12, 902)
        tokens_3d = np.stack(tokens_list, axis=1)
        B, n_tok, n_feat = tokens_3d.shape

        # Assemble flat vector: (B, 12 * 902 + 246) = (B, 11070)
        tokens_flat = tokens_3d.reshape(B, n_tok * n_feat)
        delta_2d = np.concatenate(delta_list, axis=1)
        flat_all = np.concatenate([tokens_flat, delta_2d], axis=1)

        if fit:
            flat_scaled = self.scaler.fit_transform(flat_all)
        else:
            flat_scaled = self.scaler.transform(flat_all)

        # Scale tokens proportionally from standardized flat vector
        tokens_scaled = flat_scaled[:, :n_tok * n_feat].reshape(B, n_tok, n_feat)
        return flat_scaled, tokens_scaled

    def fit(self, train_loader):
        torch.manual_seed(42)
        np.random.seed(42)

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
        flat_all, tokens_all = self._extract_disentangled_tokens(X_all, fit=True)

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
            optimizer, T_max=self.n_epochs_train, eta_min=1e-5
        )
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        self.net.train()
        for epoch in range(self.n_epochs_train):
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

        flat, tokens = self._extract_disentangled_tokens(X_np, fit=False)
        flat_tensor = torch.tensor(flat, dtype=torch.float32, device=self.device)
        tokens_tensor = torch.tensor(tokens, dtype=torch.float32, device=self.device)

        self.net.eval()
        logits = self.net(flat_tensor, tokens_tensor)
        preds = logits.argmax(dim=-1).cpu().numpy()
        return preds.astype(np.int64)


class Solver(CompetSolver):
    """Competition solver for Spatio-Temporal Deep Transformer."""

    name = "SpatioTemporalTransformer"

    requirements = ["pip::pyriemann"]

    def load_model(self, meta):
        sfreq = meta.get("sfreq", 120.0)
        model = SpatioTemporalTransformerModel(
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

            model.net = SpatioTemporalTransformerNet(
                n_bands=model.n_bands,
                n_epochs=model.n_time_epochs,
                in_features_per_token=model.in_features_per_token,
                n_delta_features=model.n_delta_features,
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

