# LGFF-Net Semantic Segmentation Reproduction

This project reproduces the core modules described in the paper for LGFF-Net-style point cloud semantic segmentation, including:

- **Adaptive Neighborhood Search (ANS)** for local neighborhood construction.
- **Local Feature Aggregation (LFA)** with geometry and semantic branches.
- **Global Feature Enhancement (GFE)** with channel and point attention.

## Project Structure

```
lgff/
  data.py          # Dataset loader for .npz point clouds
  model.py         # LGFF-Net implementation
  utils.py         # kNN and sampling utilities
scripts/
  train.py         # Training entry point
  infer.py         # Inference entry point
requirements.txt
```

## Data Format

Prepare data as `.npz` files containing:

- `points`: float32 array of shape `(N, 3)`
- `labels`: int64 array of shape `(N,)`

Organize the dataset:

```
data/
  train/
    sample_000.npz
    sample_001.npz
```

## Training

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/train.py --data-root data --num-classes 13 --epochs 200
```

## Inference

```bash
python scripts/infer.py --checkpoint checkpoints/lgff_net.pt --points demo_points.npy --num-classes 13
```

## Notes

- This is a faithful, minimal reproduction aimed at enabling end-to-end semantic segmentation experiments.
- For large point clouds, consider pre-sampling or using the `LGFFNetWithSampling` class in `lgff/model.py`.
