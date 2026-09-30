"""Ben Graham preprocessing and one-off disk caching.

Preprocessing is deterministic, so running it once and storing lossless PNGs
gives pixel-identical inputs to on-the-fly preprocessing while removing the
full-resolution decode + Gaussian blur from every training step.
"""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers


def ben_graham(im_rgb, size=512):
    """Circular crop + local mean subtraction (identical to the training code)."""
    h, w = im_rgb.shape[:2]
    radius = min(h, w) // 2
    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= radius ** 2).astype(np.uint8)
    im = cv2.resize(im_rgb, (size, size))
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    local = cv2.GaussianBlur(im, (0, 0), sigmaX=size / 30)
    im = cv2.addWeighted(im, 4, local, -4, 128)
    im[mask == 0] = 0
    return im


def read_rgb(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def resolve(csv_path, p):
    p = Path(p)
    return p if p.is_absolute() else Path(csv_path).resolve().parent / p


def _cache_one(job):
    src, dst, size = job
    if not dst.exists():
        im = ben_graham(read_rgb(src), size)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))  # PNG is lossless
    return str(dst)


def cache_csv(csv_path, out_dir, size=512, workers=8):
    """Write preprocessed PNGs and a CSV (path, grade, source, cached=1) pointing to them."""
    out_dir = Path(out_dir).resolve()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    jobs = []
    for i, r in enumerate(rows):
        src = resolve(csv_path, r["path"])
        jobs.append((src, out_dir / "images" / f"{i:06d}_{src.stem}.png", size))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        dsts = list(ex.map(_cache_one, jobs, chunksize=16))
    out_csv = out_dir / Path(csv_path).name
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "grade", "source", "cached", "size"])
        w.writeheader()
        for r, (src, _, _), dst in zip(rows, jobs, dsts):
            w.writerow(dict(path=dst, grade=r["grade"], source=str(src), cached=1, size=size))
    return out_csv
