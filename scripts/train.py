"""Train LGFF-Net for point cloud semantic segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from lgff.data import PointCloudSegDataset, collate_fn
from lgff.model import LGFFNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LGFF-Net")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = PointCloudSegDataset(args.data_root, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = LGFFNet(num_classes=args.num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            points = batch["points"].to(args.device)
            labels = batch["labels"].to(args.device)
            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}")

    ckpt_path = args.output / "lgff_net.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
