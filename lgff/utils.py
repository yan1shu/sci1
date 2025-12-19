"""Utility functions for point cloud processing."""

from __future__ import annotations

import torch


def knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """Return kNN indices for each point.

    Args:
        points: (B, N, C) tensor of point coordinates.
        k: number of neighbors.
    """
    distances = torch.cdist(points, points)
    return distances.topk(k=k, largest=False).indices


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: (B, N, C) tensor.
        idx: (B, M, K) tensor of indices.
    """
    batch_indices = torch.arange(points.size(0), device=points.device)[:, None, None]
    return points[batch_indices, idx]


def random_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Randomly sample a subset of points.

    Args:
        points: (B, N, C) tensor.
        num_samples: number of points to sample.
    """
    batch_size, num_points, _ = points.shape
    indices = torch.randint(0, num_points, (batch_size, num_samples), device=points.device)
    batch_indices = torch.arange(batch_size, device=points.device)[:, None]
    return points[batch_indices, indices]
