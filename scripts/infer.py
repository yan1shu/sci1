"""Run inference with a trained LGFF-Net model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lgff.model import LGFFNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LGFF-Net inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("predictions.npy"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = LGFFNet(num_classes=args.num_classes).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    points = np.load(args.points)
    points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(args.device)
    with torch.no_grad():
        logits = model(points_tensor)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    np.save(args.output, preds)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
