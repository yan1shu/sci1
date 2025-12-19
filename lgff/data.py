"""Dataset utilities for point cloud segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PointCloudSegDataset(Dataset):
    """Loads point clouds and semantic labels from .npz files.

    Expected .npz format with arrays: points (N, 3) and labels (N,).
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        self.files = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No npz files found in {self.root / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        points = torch.from_numpy(data["points"]).float()
        labels = torch.from_numpy(data["labels"]).long()
        return points, labels


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    points, labels = zip(*batch)
    return {
        "points": torch.stack(points, dim=0),
        "labels": torch.stack(labels, dim=0),
    }
