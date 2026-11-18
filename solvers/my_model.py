"""Custom EEG neural network solver for Track 02 BCI Decoding.

Architecture combines:
1. Per-window Z-score normalization (robust to microvolt variance & impedance drift).
2. Temporal convolution to capture oscillatory frequency bands (mu/alpha and beta).
3. Spatial depthwise convolution to isolate sensorimotor cortical dipoles across channels.
4. Activation and temporal pooling for signal power extraction.
5. Linear classification head with dropout regularization.
"""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver


class MyCustomEEGNet(nn.Module):
    """EEG feature extraction network with temporal and spatial convolutions."""

    def __init__(self, n_chans=41, n_times=480, n_classes=3, device="cpu"):
        super().__init__()
        self.device = device
        self.n_chans = n_chans
        self.n_classes = n_classes

        # Layer 1: Temporal convolution (digital frequency bandpass filter)
        # kernel size 31 at 120 Hz covers ~0.25 s (full cycle of 8-12 Hz rhythm)
        self.temporal_filter = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(1, 31),
            padding=(0, 15),
            bias=False,
        )

        # Layer 2: Spatial depthwise convolution (learns channel combinations)
        self.spatial_filter = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(n_chans, 1),
            groups=8,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(16)
        self.act = nn.ELU()

        # Layer 3: Temporal average pooling (converts wave oscillations to power)
        self.pool = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout = nn.Dropout(0.25)

        # Adaptive pool ensures fixed feature dimension regardless of window length
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))
        self.classifier = nn.Linear(16 * 8, n_classes)

    def forward(self, x):
        # Ensure tensor is float32 on the right device
        x = x.to(self.device, dtype=torch.float32)

        # Ensure 4D shape: (B, 1, C, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Window-level Z-score normalization across time for each channel
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std

        # Convolutional feature extraction
        x = self.temporal_filter(x)
        x = self.spatial_filter(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Global feature aggregation and classification
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        logits = self(X)
        return logits.argmax(dim=1)


class Solver(CompetSolver):
    """Competition solver wrapper for MyCustomEEGNet."""

    name = "MyCustomModel"

    def load_model(self, meta):
        model = MyCustomEEGNet(
            n_chans=meta["n_chans"],
            n_times=meta["n_times"],
            n_classes=meta["n_classes"],
            device=self.device,
        ).to(self.device)

        weights = meta["submission_dir"] / "weights.pt"
        if weights.exists() and self.train_loader is None:
            print(f"[loading] {weights} into {self.name}")
            model.load_state_dict(torch.load(weights, map_location=self.device))
        return model

    def fit(self, model, train_loader):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(15):
            for X, y, _info in train_loader:
                opt.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
                opt.step()

    def save_model(self, model, path):
        torch.save(model.state_dict(), path / "weights.pt")

