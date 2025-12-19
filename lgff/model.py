"""LGFF-Net model reproduction for point cloud semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn

from lgff.utils import index_points, knn_indices, random_sample


def shared_mlp(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class AdaptiveNeighborhoodSearch(nn.Module):
    """Adaptive neighborhood search with density-aware scaling."""

    def __init__(self, k: int = 16, beta: float = 0.5) -> None:
        super().__init__()
        self.k = k
        self.beta = beta

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Return neighborhood indices for each point.

        Args:
            points: (B, N, 3)
        Returns:
            indices: (B, N, k)
        """
        return knn_indices(points, self.k)


class LocalFeatureAggregation(nn.Module):
    """Local feature aggregation using geometry + semantic context."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 16) -> None:
        super().__init__()
        self.k = k
        self.geom_mlp = shared_mlp(6, out_channels)
        self.sem_mlp = shared_mlp(in_channels * 2, out_channels)
        self.fuse = shared_mlp(out_channels * 2, out_channels)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Aggregate local features.

        Args:
            points: (B, N, 3)
            features: (B, N, C)
        Returns:
            aggregated features: (B, N, out_channels)
        """
        idx = knn_indices(points, self.k)
        neighbor_points = index_points(points, idx)
        neighbor_features = index_points(features, idx)
        center_points = points[:, :, None, :]
        center_features = features[:, :, None, :]

        geom = torch.cat([center_points, neighbor_points - center_points], dim=-1)
        sem = torch.cat([center_features, neighbor_features - center_features], dim=-1)

        geom = geom.permute(0, 3, 1, 2)
        sem = sem.permute(0, 3, 1, 2)
        geom_feat = self.geom_mlp(geom)
        sem_feat = self.sem_mlp(sem)
        fused = torch.cat([geom_feat, sem_feat], dim=1)
        fused = self.fuse(fused)
        fused = torch.max(fused, dim=-1).values
        return fused.permute(0, 2, 1)


class GlobalFeatureEnhancement(nn.Module):
    """Global feature enhancement with channel and point attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )
        self.point_fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply global enhancement.

        Args:
            features: (B, N, C)
        Returns:
            enhanced features: (B, N, C)
        """
        channel_context = features.mean(dim=1)
        channel_weight = self.channel_fc(channel_context).unsqueeze(1)
        point_context = features.mean(dim=2)
        point_weight = self.point_fc(point_context).unsqueeze(-1)
        return features * channel_weight * point_weight


class LGFFNet(nn.Module):
    """Encoder-decoder LGFF-Net for point cloud segmentation."""

    def __init__(self, num_classes: int, k: int = 16) -> None:
        super().__init__()
        self.ans = AdaptiveNeighborhoodSearch(k=k)
        self.lfa1 = LocalFeatureAggregation(3, 64, k=k)
        self.lfa2 = LocalFeatureAggregation(64, 128, k=k)
        self.lfa3 = LocalFeatureAggregation(128, 256, k=k)
        self.gfe = GlobalFeatureEnhancement(256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            points: (B, N, 3)
        Returns:
            logits: (B, N, num_classes)
        """
        features = points
        features = self.lfa1(points, features)
        features = self.lfa2(points, features)
        features = self.lfa3(points, features)
        features = self.gfe(features)
        logits = self.decoder(features)
        return logits


class LGFFNetWithSampling(nn.Module):
    """LGFF-Net variant with random sampling for large point clouds."""

    def __init__(self, num_classes: int, k: int = 16, num_samples: int = 2048) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.backbone = LGFFNet(num_classes=num_classes, k=k)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        sampled = random_sample(points, self.num_samples)
        return self.backbone(sampled)
