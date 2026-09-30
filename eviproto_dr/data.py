import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T

from .preprocess import ben_graham, read_rgb, resolve

NUM_GRADES = 5
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class FundusDataset(Dataset):
    """Reads a CSV with columns path,grade (optional: source,cached,size).

    Rows produced by `run.py preprocess` carry cached=1 and are read directly.
    Items may be an int index or an (index, aug_seed) pair. With a seed, the
    augmentation draw depends only on (seed, epoch, batch position), so results
    do not change with the number of DataLoader workers or after a resume.
    """

    def __init__(self, csv_path, train=False, size=512):
        with open(csv_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"{csv_path} is empty")
        self.csv_path, self.size = csv_path, size
        self.labels = [int(r["grade"]) for r in self.rows]
        bad = sorted({k for k in self.labels if k not in range(NUM_GRADES)})
        if bad:
            raise ValueError(f"{csv_path}: grades must be 0..4, found {bad} "
                             "(remove ungradable images, e.g. DDR grade 5)")
        for r in self.rows:
            if r.get("cached") == "1" and int(r.get("size", size)) != size:
                raise ValueError(f"Cache built at size {r['size']}, dataset expects {size}")
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15),
               T.ColorJitter(.15, .15, .15, .05), T.RandomApply([T.GaussianBlur(3)], p=.2)] if train else []
        self.augment = T.Compose(aug) if aug else None
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        idx, seed = item if isinstance(item, tuple) else (item, None)
        row = self.rows[idx]
        path = resolve(self.csv_path, row["path"])
        im = read_rgb(path)
        if row.get("cached") != "1":
            im = ben_graham(im, self.size)
        img = Image.fromarray(im)
        if self.augment is not None:
            if seed is None:
                img = self.augment(img)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    img = self.augment(img)
        return self.to_tensor(img), self.labels[idx], row.get("source") or str(path)


class BalancedBatchSampler(Sampler):
    """Grade-aware sampling with replacement: every batch holds >= min_per_grade of each grade."""

    def __init__(self, labels, batch_size=32, min_per_grade=4, seed=42):
        self.groups = [np.flatnonzero(np.asarray(labels) == k) for k in range(NUM_GRADES)]
        if any(len(g) == 0 for g in self.groups):
            raise ValueError("Training data needs all five grades")
        if batch_size < NUM_GRADES * min_per_grade:
            raise ValueError(f"Batch size must be >= {NUM_GRADES * min_per_grade}")
        self.all = np.concatenate(self.groups)
        self.batch_size, self.minimum, self.seed = batch_size, min_per_grade, seed
        self.epoch = 0
        self.length = (len(labels) + batch_size - 1) // batch_size

    def set_epoch(self, epoch_index):
        """epoch_index is 0-based; call before every epoch (also after resume)."""
        self.epoch = epoch_index

    def __len__(self):
        return self.length

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        aug_rng = np.random.default_rng([self.seed, self.epoch, 7])
        for _ in range(self.length):
            picks = [rng.choice(g, self.minimum, replace=True) for g in self.groups]
            picks.append(rng.choice(self.all, self.batch_size - NUM_GRADES * self.minimum, replace=True))
            b = np.concatenate(picks)
            rng.shuffle(b)
            seeds = aug_rng.integers(0, 2**31 - 1, size=len(b))
            yield [(int(i), int(s)) for i, s in zip(b, seeds)]
