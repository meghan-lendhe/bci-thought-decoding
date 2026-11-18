"""Adaptive Master Ensemble Solver with Online Riemannian Alignment.

Combines:
1. Spatio-Temporal Disentangled Token Backbone:
   - 6 neurophysiological bands (Theta, Low Alpha, High Alpha, Low Beta, High Beta, Low Gamma).
   - Early (1-3s) vs. Late (3-5s) temporal epochs -> 12 tokens.
   - Disentangles Log-Power (41) and Tangent Space Correlation (861) -> 902 features per token.
   - Explicit Delta-Power trajectory features (246 features).
   - Total flat feature dimension: 11,070 features.

2. Online Riemannian Alignment (RA):
   - Pre-whitens training covariance matrices to Identity Matrix I.
   - Exposes causal reset_state() at each session start.
   - Tracks running baseline covariance per band/epoch during streamed inference.
   - Pre-whitening incoming test windows with trace-regularized inverse square root:
     C_tilde = C_ref^{-1/2} C C_ref^{-1/2} -> centered at I with zero domain gap!

3. Dual Champion Decision Heads:
   - Head 1: Ridge Classifier with L2 shrinkage (alpha=150.0).
   - Head 2: SpatioTemporalTransformer with Multi-Head Self-Attention.
   - Blended ensemble: P = 0.55 * P_ridge + 0.45 * P_transformer.
"""

import copy
import joblib
import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


def _invsqrtm(C, eps=1e-5):
    """Numerically stable inverse matrix square root with trace shrinkage."""
    trace = np.trace(C) / len(C)
    C_reg = 0.95 * C + 0.05 * trace * np.eye(len(C))
    eigvals, eigvecs = np.linalg.eigh(C_reg)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


class SpatioTemporalTransformerNet(nn.Module):
    """Wide-and-Deep Spatio-Temporal Transformer Architecture."""

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
        self.n_delta_features = n_delta_features
        self.d_model = d_model
        self.n_classes = n_classes

        # Token input projection
        self.token_proj = nn.Sequential(
            nn.Linear(in_features_per_token, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Spatio-Temporal Learnable Positional Encoding
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)

        # Multi-Head Attention Encoder
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

        # Global Highway: Direct linear projection from all flat features
        self.highway = nn.Sequential(
            nn.Linear(self.n_tokens * in_features_per_token + n_delta_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, flat_features, tokens):
        B = tokens.size(0)

        # 1. Project tokens into d_model embedding space
        h = self.token_proj(tokens)  # (B, 12, d_model)
        h = h + self.pos_emb

        # 2. Multi-Head Self-Attention across frequency bands & time
        h_trans = self.transformer(h)  # (B, 12, d_model)

        # 3. Dual Pooling (Mean + Max across tokens)
        mean_pool = h_trans.mean(dim=1)  # (B, d_model)
        max_pool = h_trans.max(dim=1).values  # (B, d_model)

        # 4. Global Feature Highway
        highway_feat = self.highway(flat_features)  # (B, d_model)

        # 5. Concatenate and classify
        combined = torch.cat([mean_pool, max_pool, highway_feat], dim=-1)
        return self.classifier(combined)


class AdaptiveMasterEnsembleModel:
    """Master Ensemble with Online Riemannian Alignment (RA) and reset_state()."""

    def __init__(
        self,
        n_chans=41,
        n_classes=3,
        sfreq=120.0,
        alpha=150.0,
        d_model=128,
        n_heads=4,
        n_layers=1,
        dropout=0.50,
        n_epochs_train=18,
        lr=1e-3,
        weight_decay=1e-3,
        label_smoothing=0.10,
        batch_size=16,
        w_ridge=0.55,
        w_trans=0.45,
        device="cpu",
    ):
        self.n_chans = int(n_chans)
        self.n_classes = int(n_classes)
        self.sfreq = float(sfreq)
        self.alpha = float(alpha)
        self.device = device

        # 6 Neurophysiological bands
        self.bands = [
            (4.0, 8.0),    # Theta (Mental Subtraction)
            (8.0, 11.0),   # Low Alpha (Sensory gating)
            (11.0, 14.0),  # High Alpha / Mu (Motor Imagery)
            (14.0, 20.0),  # Low Beta (Motor planning)
            (20.0, 28.0),  # High Beta (Active execution)
            (28.0, 40.0),  # Low Gamma (Word Generation)
        ]
        self.n_bands = len(self.bands)
        self.n_time_epochs = 2  # Early (1-3s) and Late (3-5s)
        self.n_tokens = self.n_bands * self.n_time_epochs  # 12 tokens

        self.tangent_dim = (self.n_chans * (self.n_chans + 1)) // 2  # 861
        self.in_features_per_token = self.n_chans + self.tangent_dim   # 902
        self.n_delta_features = self.n_bands * self.n_chans           # 246
        self.total_flat_features = (self.n_tokens * self.in_features_per_token) + self.n_delta_features

        # Estimators
        self.cov_early = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_early = [TangentSpace(metric="riemann") for _ in self.bands]
        self.cov_late = [Covariances(estimator="lwf") for _ in self.bands]
        self.ts_late = [TangentSpace(metric="riemann") for _ in self.bands]

        # Training baselines (stored for RA)
        self.ref_e_train = []
        self.ref_l_train = []
        self.R_tr_e_inv = []
        self.R_tr_l_inv = []

        self.scaler = StandardScaler()
        self.ridge = RidgeClassifier(alpha=self.alpha)

        # Transformer Head
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

        self.w_ridge = float(w_ridge)
        self.w_trans = float(w_trans)
        self.fitted = False

        # Online streaming state
        self.reset_state()

    def reset_state(self):
        """Called automatically at the beginning of each test session by fresh_model()."""
        self.running_cov_e = [np.zeros((self.n_chans, self.n_chans)) for _ in range(self.n_bands)]
        self.running_cov_l = [np.zeros((self.n_chans, self.n_chans)) for _ in range(self.n_bands)]
        self.n_test_windows = 0

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

    def fit(self, train_loader):
        torch.manual_seed(42)
        np.random.seed(42)

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

        split_idx = X_all.shape[-1] // 2
        X_early = X_all[..., :split_idx]
        X_late = X_all[..., split_idx:]

        # Step 1: Compute training covariances & Whitening operators
        self.ref_e_train = []
        self.ref_l_train = []
        self.R_tr_e_inv = []
        self.R_tr_l_inv = []

        tokens_list = []
        delta_list = []

        for i, (f_low, f_high) in enumerate(self.bands):
            X_e_b = self._filter_band(X_early, f_low, f_high)
            X_l_b = self._filter_band(X_late, f_low, f_high)

            cov_e = self.cov_early[i].fit_transform(X_e_b)
            cov_l = self.cov_late[i].fit_transform(X_l_b)

            ref_e = np.mean(cov_e, axis=0)
            ref_l = np.mean(cov_l, axis=0)
            self.ref_e_train.append(ref_e)
            self.ref_l_train.append(ref_l)

            inv_e = _invsqrtm(ref_e)
            inv_l = _invsqrtm(ref_l)
            self.R_tr_e_inv.append(inv_e)
            self.R_tr_l_inv.append(inv_l)

            # Pre-whiten training data to Identity I
            cov_e_aligned = np.array([inv_e @ c @ inv_e for c in cov_e])
            cov_l_aligned = np.array([inv_l @ c @ inv_l for c in cov_l])

            corr_e, diag_e = self._cov_to_corr(cov_e_aligned)
            conn_e = self.ts_early[i].fit_transform(corr_e)

            corr_l, diag_l = self._cov_to_corr(cov_l_aligned)
            conn_l = self.ts_late[i].fit_transform(corr_l)

            p_early = np.log(np.maximum(diag_e, 1e-12))
            p_late = np.log(np.maximum(diag_l, 1e-12))
            p_delta = p_late - p_early

            tokens_list.append(np.concatenate([p_early, conn_e], axis=1))
            tokens_list.append(np.concatenate([p_late, conn_l], axis=1))
            delta_list.append(p_delta)

        tokens_3d = np.stack(tokens_list, axis=1)
        B, n_tok, n_feat = tokens_3d.shape
        tokens_flat = tokens_3d.reshape(B, n_tok * n_feat)
        delta_2d = np.concatenate(delta_list, axis=1)
        flat_all = np.concatenate([tokens_flat, delta_2d], axis=1)

        flat_scaled = self.scaler.fit_transform(flat_all)
        tokens_scaled = flat_scaled[:, :n_tok * n_feat].reshape(B, n_tok, n_feat)

        # Step 2: Fit Head 1 (Ridge)
        self.ridge.fit(flat_scaled, y_all)

        # Step 3: Fit Head 2 (Transformer)
        dataset = TensorDataset(
            torch.tensor(flat_scaled, dtype=torch.float32),
            torch.tensor(tokens_scaled, dtype=torch.float32),
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

        N_windows = len(X_np)
        split_idx = X_np.shape[-1] // 2
        X_early = X_np[..., :split_idx]
        X_late = X_np[..., split_idx:]

        all_preds = []

        # Process each window causally (streamed B=1)
        for w_idx in range(N_windows):
            self.n_test_windows += 1
            w_e = X_early[w_idx:w_idx+1]
            w_l = X_late[w_idx:w_idx+1]

            tokens_curr = []
            delta_curr = []

            for i, (f_low, f_high) in enumerate(self.bands):
                X_e_b = self._filter_band(w_e, f_low, f_high)
                X_l_b = self._filter_band(w_l, f_low, f_high)

                c_e = self.cov_early[i].transform(X_e_b)[0]
                c_l = self.cov_late[i].transform(X_l_b)[0]

                # Update online running baseline accumulator
                self.running_cov_e[i] += c_e
                self.running_cov_l[i] += c_l

                # Online running baseline mean
                ref_e_test = self.running_cov_e[i] / self.n_test_windows
                ref_l_test = self.running_cov_l[i] / self.n_test_windows

                # Whitening operators
                inv_e = _invsqrtm(ref_e_test)
                inv_l = _invsqrtm(ref_l_test)

                # Pre-whiten incoming window to Identity I
                c_e_aligned = (inv_e @ c_e @ inv_e)[np.newaxis, ...]
                c_l_aligned = (inv_l @ c_l @ inv_l)[np.newaxis, ...]

                corr_e, diag_e = self._cov_to_corr(c_e_aligned)
                conn_e = self.ts_early[i].transform(corr_e)

                corr_l, diag_l = self._cov_to_corr(c_l_aligned)
                conn_l = self.ts_late[i].transform(corr_l)

                p_e = np.log(np.maximum(diag_e, 1e-12))
                p_l = np.log(np.maximum(diag_l, 1e-12))
                p_delta = p_l - p_e

                tokens_curr.append(np.concatenate([p_e, conn_e], axis=1))
                tokens_curr.append(np.concatenate([p_l, conn_l], axis=1))
                delta_curr.append(p_delta)

            tokens_3d = np.stack(tokens_curr, axis=1)  # (1, 12, 902)
            tokens_flat = tokens_3d.reshape(1, -1)
            delta_2d = np.concatenate(delta_curr, axis=1)
            flat_curr = np.concatenate([tokens_flat, delta_2d], axis=1)

            flat_scaled = self.scaler.transform(flat_curr)
            tokens_scaled = flat_scaled[:, :self.n_tokens * self.in_features_per_token].reshape(1, self.n_tokens, -1)

            # Head 1 (Ridge)
            ridge_margins = self.ridge.decision_function(flat_scaled)
            p_ridge = softmax(ridge_margins, axis=-1)

            # Head 2 (Transformer)
            flat_tensor = torch.tensor(flat_scaled, dtype=torch.float32, device=self.device)
            tokens_tensor = torch.tensor(tokens_scaled, dtype=torch.float32, device=self.device)
            self.net.eval()
            trans_logits = self.net(flat_tensor, tokens_tensor)
            p_trans = torch.softmax(trans_logits, dim=-1).cpu().numpy()

            # Ensemble Blend
            p_ensemble = self.w_ridge * p_ridge + self.w_trans * p_trans
            all_preds.append(p_ensemble.argmax(axis=-1)[0])

        return np.array(all_preds, dtype=np.int64)


class Solver(CompetSolver):
    """Competition solver for Adaptive Master Ensemble with Online Riemannian Alignment."""

    name = "AdaptiveMasterEnsemble"

    requirements = ["pip::pyriemann"]

    def load_model(self, meta):
        device = meta.get("device", getattr(self, "device", "cpu"))
        sfreq = meta.get("sfreq", 120.0)
        model = AdaptiveMasterEnsembleModel(
            n_chans=meta["n_chans"],
            n_classes=meta["n_classes"],
            sfreq=sfreq,
            device=device,
        )

        weights_joblib = meta["submission_dir"] / "weights.joblib"
        weights_pt = meta["submission_dir"] / "weights.pt"

        if weights_joblib.exists() and getattr(self, "train_loader", None) is None:
            print(f"[loading] {weights_joblib} into {self.name}")
            artifact = joblib.load(weights_joblib)

            model.cov_early = artifact["cov_early"]
            model.ts_early = artifact["ts_early"]
            model.cov_late = artifact["cov_late"]
            model.ts_late = artifact["ts_late"]
            model.ref_e_train = artifact.get("ref_e_train", [])
            model.ref_l_train = artifact.get("ref_l_train", [])
            model.scaler = artifact["scaler"]
            model.bands = artifact["bands"]
            model.sfreq = artifact["sfreq"]
            model.n_chans = artifact["n_chans"]
            model.n_classes = artifact["n_classes"]
            model.n_bands = len(model.bands)
            model.ridge = artifact["ridge"]
            model.w_ridge = artifact.get("w_ridge", 0.55)
            model.w_trans = artifact.get("w_trans", 0.45)

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
            ).to(device)

            if weights_pt.exists():
                net_weights = torch.load(weights_pt, map_location=device, weights_only=True)
                model.net.load_state_dict(net_weights)
            elif "net_state_dict" in artifact:
                model.net.load_state_dict(artifact["net_state_dict"])

            model.net.to(device)
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
            "ref_e_train": model.ref_e_train,
            "ref_l_train": model.ref_l_train,
            "scaler": model.scaler,
            "bands": model.bands,
            "sfreq": model.sfreq,
            "n_chans": model.n_chans,
            "n_classes": model.n_classes,
            "ridge": model.ridge,
            "w_ridge": model.w_ridge,
            "w_trans": model.w_trans,
            "net_state_dict": {k: v.cpu() for k, v in model.net.state_dict().items()},
        }
        joblib.dump(artifact, path / "weights.joblib")
        torch.save(model.net.state_dict(), path / "weights.pt")

