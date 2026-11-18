"""Adaptive Regional Master Ensemble Solver.

Combines:
1. Neuroanatomical Regional Subsystem Backbone (3,342 clean features):
   - Language Subsystem (10 channels): F3, F5, F1, FC3, FC5, AF3, TP7, CP5, P3, Fz (Broca & Wernicke)
   - Sensorimotor Subsystem (11 channels): C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4, FCz (Motor Strip)
   - Calculation Subsystem (14 channels): Pz, P1, P2, P3, P4, P5, P6, PO7, PO8, CPz, Fz, FCz, F1, F2 (Parietal IPS)
   - Excludes 10 visual/peripheral noise channels from corrupting the representations.
   - 6 Bands x 2 Epochs x 3 Subsystems = 36 Spatio-Temporal Tokens (3,132 token features + 210 delta features).

2. Bayesian Warm-Start Online Riemannian Alignment (N0 = 15):
   - Population baseline prior eliminates cold-start signal erasure on trial 1.
   - Causal streaming updates dynamically adapt to the test subject's personal geometry.

3. Dual Champion Decision Heads:
   - Head 1: Regional Ridge with L2 shrinkage (alpha=750.0).
   - Head 2: Regional SpatioTemporal Transformer with independent LayerNorm per subsystem.
   - Blended Ensemble: P = 0.50 * P_ridge + 0.50 * P_transformer.
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


def _cov_to_corr(covs):
    diags = np.diagonal(covs, axis1=-2, axis2=-1)
    stds = np.sqrt(np.maximum(diags, 1e-12))[..., np.newaxis]
    corrs = covs / (stds @ stds.swapaxes(-2, -1))
    for i in range(len(corrs)):
        np.fill_diagonal(corrs[i], 1.0)
    return corrs, diags


class RegionalTransformerNet(nn.Module):
    """Spatio-Temporal Regional Transformer with Subsystem-Specific LayerNorm."""

    def __init__(
        self,
        dim_lang=65,
        dim_motor=77,
        dim_math=119,
        n_flat_features=3342,
        d_model=128,
        n_heads=4,
        dim_feedforward=256,
        n_layers=1,
        dropout=0.30,
        n_classes=3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = 36  # 6 bands x 2 epochs x 3 ROIs

        # Subsystem-specific projections (rescales all 3 subsystems to equal variance)
        self.proj_lang = nn.Sequential(
            nn.Linear(dim_lang, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_motor = nn.Sequential(
            nn.Linear(dim_motor, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_math = nn.Sequential(
            nn.Linear(dim_math, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable Spatio-Temporal Positional Embedding
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)

        # Multi-Head Attention Encoder across all 36 tokens
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

        # Global Highway from all 3,342 flat features
        self.highway = nn.Sequential(
            nn.Linear(n_flat_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, flat_features, tokens_lang, tokens_motor, tokens_math):
        # 1. Project tokens into shared d_model embedding
        h_lang = self.proj_lang(tokens_lang)      # (B, 12, d_model)
        h_motor = self.proj_motor(tokens_motor)   # (B, 12, d_model)
        h_math = self.proj_math(tokens_math)      # (B, 12, d_model)

        # 2. Concatenate along token sequence dimension -> (B, 36, d_model)
        h = torch.cat([h_lang, h_motor, h_math], dim=1) + self.pos_emb

        # 3. Multi-Head Attention across frequency, time, and brain regions
        h_trans = self.transformer(h)

        # 4. Dual Pooling
        mean_pool = h_trans.mean(dim=1)
        max_pool = h_trans.max(dim=1).values

        # 5. Global Feature Highway
        h_high = self.highway(flat_features)

        # 6. Classify
        combined = torch.cat([mean_pool, max_pool, h_high], dim=-1)
        return self.classifier(combined)


class AdaptiveRegionalEnsembleModel:
    """Regional Subsystem Model with Bayesian Warm-Start Alignment."""

    def __init__(
        self,
        n_chans=41,
        n_classes=3,
        sfreq=120.0,
        alpha=750.0,
        prior_N0=15,
        d_model=128,
        n_heads=4,
        n_layers=1,
        dropout=0.30,
        n_epochs_train=20,
        lr=1e-3,
        weight_decay=1e-3,
        label_smoothing=0.10,
        batch_size=16,
        w_ridge=0.50,
        w_trans=0.50,
        device="cpu",
    ):
        self.n_chans = int(n_chans)
        self.n_classes = int(n_classes)
        self.sfreq = float(sfreq)
        self.alpha = float(alpha)
        self.prior_N0 = int(prior_N0)
        self.device = device

        self.ch_names = "C2,CP2,P2,Pz,CPz,Cz,C1,CP1,P1,P3,PO7,P5,TP7,CP5,CP3,C3,PO8,P6,P4,TP8,CP6,CP4,C6,C4,FC6,FC4,FC2,F6,F4,F2,AF4,AF3,F5,F3,F1,Fz,FC5,FC3,FC1,FCz,C5".split(",")

        # 6 Neurophysiological bands
        self.bands = [
            (4.0, 8.0),    # Theta
            (8.0, 11.0),   # Low Alpha
            (11.0, 14.0),  # High Alpha
            (14.0, 20.0),  # Low Beta
            (20.0, 28.0),  # High Beta
            (28.0, 40.0),  # Low Gamma
        ]
        self.n_bands = len(self.bands)

        # 3 Brain Subsystems
        self.roi_names = ["Language", "Motor", "Calculation"]
        self.roi_channel_lists = [
            ["F3", "F5", "F1", "FC3", "FC5", "AF3", "TP7", "CP5", "P3", "Fz"],                              # Lang (10)
            ["C3", "C1", "Cz", "C2", "C4", "CP3", "CP1", "CPz", "CP2", "CP4", "FCz"],                       # Motor (11)
            ["Pz", "P1", "P2", "P3", "P4", "P5", "P6", "PO7", "PO8", "CPz", "Fz", "FCz", "F1", "F2"],   # Math (14)
        ]
        self.roi_indices = [
            [self.ch_names.index(ch) for ch in r_list if ch in self.ch_names]
            for r_list in self.roi_channel_lists
        ]

        self.dim_lang = len(self.roi_indices[0]) + (len(self.roi_indices[0]) * (len(self.roi_indices[0]) + 1)) // 2  # 10 + 55 = 65
        self.dim_motor = len(self.roi_indices[1]) + (len(self.roi_indices[1]) * (len(self.roi_indices[1]) + 1)) // 2 # 11 + 66 = 77
        self.dim_math = len(self.roi_indices[2]) + (len(self.roi_indices[2]) * (len(self.roi_indices[2]) + 1)) // 2   # 14 + 105 = 119
        self.total_flat_features = (6 * 2 * (self.dim_lang + self.dim_motor + self.dim_math)) + (6 * (10 + 11 + 14)) # 3342

        # PyRiemann Estimators
        self.cov_e = [[Covariances(estimator="lwf") for _ in range(3)] for _ in range(self.n_bands)]
        self.cov_l = [[Covariances(estimator="lwf") for _ in range(3)] for _ in range(self.n_bands)]
        self.ts_e = [[TangentSpace(metric="riemann") for _ in range(3)] for _ in range(self.n_bands)]
        self.ts_l = [[TangentSpace(metric="riemann") for _ in range(3)] for _ in range(self.n_bands)]

        # Training baselines
        self.ref_e_train = [[None for _ in range(3)] for _ in range(self.n_bands)]
        self.ref_l_train = [[None for _ in range(3)] for _ in range(self.n_bands)]
        self.R_tr_e_inv = [[None for _ in range(3)] for _ in range(self.n_bands)]
        self.R_tr_l_inv = [[None for _ in range(3)] for _ in range(self.n_bands)]

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

        self.net = RegionalTransformerNet(
            dim_lang=self.dim_lang,
            dim_motor=self.dim_motor,
            dim_math=self.dim_math,
            n_flat_features=self.total_flat_features,
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

        self.reset_state()

    def reset_state(self):
        """Causal session reset with Bayesian Warm-Start Prior."""
        if hasattr(self, "ref_e_train") and self.ref_e_train[0][0] is not None:
            self.running_cov_e = [
                [self.prior_N0 * self.ref_e_train[b][r].copy() for r in range(3)]
                for b in range(self.n_bands)
            ]
            self.running_cov_l = [
                [self.prior_N0 * self.ref_l_train[b][r].copy() for r in range(3)]
                for b in range(self.n_bands)
            ]
            self.n_test_windows = self.prior_N0
        else:
            self.running_cov_e = [
                [np.zeros((len(self.roi_indices[r]), len(self.roi_indices[r]))) for r in range(3)]
                for b in range(self.n_bands)
            ]
            self.running_cov_l = [
                [np.zeros((len(self.roi_indices[r]), len(self.roi_indices[r]))) for r in range(3)]
                for b in range(self.n_bands)
            ]
            self.n_test_windows = 0

    def _filter_band(self, X, f_low, f_high):
        sos = butter(4, [f_low, f_high], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)

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
        X_e = X_all[..., :split_idx]
        X_l = X_all[..., split_idx:]

        tokens_lang_list, tokens_motor_list, tokens_math_list = [], [], []
        all_tokens_list = []
        delta_list = []

        print(f"[AdaptiveRegionalEnsemble] Extracting 3,342 features for {len(X_all)} trials...")

        for b, (fl, fh) in enumerate(self.bands):
            xe_b = self._filter_band(X_e, fl, fh)
            xl_b = self._filter_band(X_l, fl, fh)

            for r, r_idx in enumerate(self.roi_indices):
                xe_roi = xe_b[:, r_idx]
                xl_roi = xl_b[:, r_idx]

                ce = self.cov_e[b][r].fit_transform(xe_roi)
                cl = self.cov_l[b][r].fit_transform(xl_roi)

                ref_e = np.mean(ce, axis=0)
                ref_l = np.mean(cl, axis=0)
                self.ref_e_train[b][r] = ref_e
                self.ref_l_train[b][r] = ref_l

                inv_e = _invsqrtm(ref_e)
                inv_l = _invsqrtm(ref_l)
                self.R_tr_e_inv[b][r] = inv_e
                self.R_tr_l_inv[b][r] = inv_l

                ce_al = np.array([inv_e @ c @ inv_e for c in ce])
                cl_al = np.array([inv_l @ c @ inv_l for c in cl])

                corr_e, de = _cov_to_corr(ce_al)
                corr_l, dl = _cov_to_corr(cl_al)

                conn_e = self.ts_e[b][r].fit_transform(corr_e)
                conn_l = self.ts_l[b][r].fit_transform(corr_l)

                pe = np.log(np.maximum(de, 1e-12))
                pl = np.log(np.maximum(dl, 1e-12))

                tok_e = np.concatenate([pe, conn_e], axis=1)
                tok_l = np.concatenate([pl, conn_l], axis=1)

                all_tokens_list.append(tok_e)
                all_tokens_list.append(tok_l)
                delta_list.append(pl - pe)

                if r == 0:
                    tokens_lang_list.extend([tok_e, tok_l])
                elif r == 1:
                    tokens_motor_list.extend([tok_e, tok_l])
                else:
                    tokens_math_list.extend([tok_e, tok_l])

        flat_tokens = np.concatenate(all_tokens_list, axis=1)
        flat_deltas = np.concatenate(delta_list, axis=1)
        flat_all = np.concatenate([flat_tokens, flat_deltas], axis=1)

        flat_scaled = self.scaler.fit_transform(flat_all)
        self.X_train_scaled = flat_scaled
        self.y_train = y_all

        # Step 2: Fit Head 1 (Ridge)
        print(f"      Fitting Head 1: Regional Ridge (alpha={self.alpha})...")
        self.ridge.fit(flat_scaled, y_all)

        # Step 3: Prepare Tokens for Head 2 (Transformer)
        B = len(flat_all)
        # 12 tokens each
        t_lang = np.stack(tokens_lang_list, axis=1)   # (B, 12, dim_lang)
        t_motor = np.stack(tokens_motor_list, axis=1) # (B, 12, dim_motor)
        t_math = np.stack(tokens_math_list, axis=1)   # (B, 12, dim_math)

        dataset = TensorDataset(
            torch.tensor(flat_scaled, dtype=torch.float32),
            torch.tensor(t_lang, dtype=torch.float32),
            torch.tensor(t_motor, dtype=torch.float32),
            torch.tensor(t_math, dtype=torch.float32),
            torch.tensor(y_all, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"      Training Head 2: Spatio-Temporal Regional Transformer ({self.n_epochs_train} epochs)...")
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs_train, eta_min=1e-5
        )
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        self.net.train()
        for epoch in range(self.n_epochs_train):
            total_loss = 0.0
            for f_b, t_lg, t_mt, t_mh, y_b in loader:
                f_b = f_b.to(self.device)
                t_lg = t_lg.to(self.device)
                t_mt = t_mt.to(self.device)
                t_mh = t_mh.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                logits = self.net(f_b, t_lg, t_mt, t_mh)
                loss = loss_fn(logits, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

        print("      [OK] Training complete.")
        self.fitted = True

    def predict(self, X):
        X_np = to_numpy(X)
        if X_np.ndim == 2:
            X_np = X_np[np.newaxis, ...]

        all_preds = []
        for w in X_np:
            self.n_test_windows += 1
            split = w.shape[-1] // 2
            w_e = w[np.newaxis, ..., :split]
            w_l = w[np.newaxis, ..., split:]

            tok_lang_curr, tok_motor_curr, tok_math_curr = [], [], []
            all_tok_curr = []
            delta_curr = []

            for b, (fl, fh) in enumerate(self.bands):
                xe_b = self._filter_band(w_e, fl, fh)
                xl_b = self._filter_band(w_l, fl, fh)

                for r, r_idx in enumerate(self.roi_indices):
                    xe_roi = xe_b[:, r_idx]
                    xl_roi = xl_b[:, r_idx]

                    ce = self.cov_e[b][r].transform(xe_roi)[0]
                    cl = self.cov_l[b][r].transform(xl_roi)[0]

                    self.running_cov_e[b][r] += ce
                    self.running_cov_l[b][r] += cl

                    ref_e_curr = self.running_cov_e[b][r] / self.n_test_windows
                    ref_l_curr = self.running_cov_l[b][r] / self.n_test_windows

                    inv_e = _invsqrtm(ref_e_curr)
                    inv_l = _invsqrtm(ref_l_curr)

                    ce_al = (inv_e @ ce @ inv_e)[np.newaxis, ...]
                    cl_al = (inv_l @ cl @ inv_l)[np.newaxis, ...]

                    corr_e, de = _cov_to_corr(ce_al)
                    corr_l, dl = _cov_to_corr(cl_al)

                    conn_e = self.ts_e[b][r].transform(corr_e)
                    conn_l = self.ts_l[b][r].transform(corr_l)

                    pe = np.log(np.maximum(de, 1e-12))
                    pl = np.log(np.maximum(dl, 1e-12))

                    tok_e = np.concatenate([pe, conn_e], axis=1)
                    tok_l = np.concatenate([pl, conn_l], axis=1)

                    all_tok_curr.extend([tok_e, tok_l])
                    delta_curr.append(pl - pe)

                    if r == 0:
                        tok_lang_curr.extend([tok_e, tok_l])
                    elif r == 1:
                        tok_motor_curr.extend([tok_e, tok_l])
                    else:
                        tok_math_curr.extend([tok_e, tok_l])

            flat_tok = np.concatenate(all_tok_curr, axis=1)
            flat_del = np.concatenate(delta_curr, axis=1)
            flat_all = np.concatenate([flat_tok, flat_del], axis=1)
            flat_sc = self.scaler.transform(flat_all)

            # Head 1 (Ridge)
            dec_ridge = self.ridge.decision_function(flat_sc)
            p_ridge = softmax(dec_ridge, axis=-1)

            # Head 2 (Transformer)
            t_lg = torch.tensor(np.stack(tok_lang_curr, axis=1), dtype=torch.float32, device=self.device)
            t_mt = torch.tensor(np.stack(tok_motor_curr, axis=1), dtype=torch.float32, device=self.device)
            t_mh = torch.tensor(np.stack(tok_math_curr, axis=1), dtype=torch.float32, device=self.device)
            f_sc = torch.tensor(flat_sc, dtype=torch.float32, device=self.device)

            self.net.eval()
            with torch.no_grad():
                logits_trans = self.net(f_sc, t_lg, t_mt, t_mh)
                p_trans = torch.softmax(logits_trans, dim=-1).cpu().numpy()

            # Ensemble Blend
            p_ens = self.w_ridge * p_ridge + self.w_trans * p_trans
            all_preds.append(p_ens.argmax(axis=-1)[0])

        return np.array(all_preds, dtype=np.int64)


class Solver(CompetSolver):
    """Competition solver for Adaptive Regional Ensemble with Bayesian Warm-Start."""

    name = "AdaptiveRegionalEnsemble"
    requirements = ["pip::pyriemann"]

    def load_model(self, meta):
        device = meta.get("device", getattr(self, "device", "cpu"))
        sfreq = meta.get("sfreq", 120.0)
        model = AdaptiveRegionalEnsembleModel(
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

            model.cov_e = artifact["cov_e"]
            model.cov_l = artifact["cov_l"]
            model.ts_e = artifact["ts_e"]
            model.ts_l = artifact["ts_l"]
            model.ref_e_train = artifact["ref_e_train"]
            model.ref_l_train = artifact["ref_l_train"]
            model.scaler = artifact["scaler"]
            model.ridge = artifact["ridge"]
            model.w_ridge = artifact.get("w_ridge", 0.50)
            model.w_trans = artifact.get("w_trans", 0.50)
            model.prior_N0 = artifact.get("prior_N0", 15)

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
            "cov_e": model.cov_e,
            "cov_l": model.cov_l,
            "ts_e": model.ts_e,
            "ts_l": model.ts_l,
            "ref_e_train": model.ref_e_train,
            "ref_l_train": model.ref_l_train,
            "scaler": model.scaler,
            "ridge": model.ridge,
            "w_ridge": model.w_ridge,
            "w_trans": model.w_trans,
            "prior_N0": model.prior_N0,
            "sfreq": model.sfreq,
            "n_chans": model.n_chans,
            "n_classes": model.n_classes,
        }
        joblib.dump(artifact, path / "weights.joblib")
        torch.save(model.net.state_dict(), path / "weights.pt")
