"""Parameter count, FLOPs, latency and memory (Tables 5-6)."""
import platform
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .preprocess import ben_graham


def environment():
    import cv2, sklearn, scipy, torchvision
    env = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
               torchvision=torchvision.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
               scipy=scipy.__version__, opencv=cv2.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name()
    return env


def count_params(model):
    groups = {name: sum(p.numel() for p in mod.parameters())
              for name, mod in model.named_children()}
    groups["total"] = sum(p.numel() for p in model.parameters())
    groups["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return groups


@torch.no_grad()
def flops(model, size, device):
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    with FlopCounterMode(display=False) as fc:
        model(x)
    total = fc.get_total_flops()
    return dict(gflops=total / 1e9, gmacs=total / 2e9)


@torch.no_grad()
def latency(model, size, device, iters=200, warmup=20, include_preprocessing=False, raw_hw=(2136, 3216)):
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    raw = np.random.randint(0, 255, (*raw_hw, 3), dtype=np.uint8)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        model(x)
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        if include_preprocessing:
            im = ben_graham(raw, size)
            x = torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(device)
        model(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(times)
    return dict(ms_mean=float(t.mean()), ms_median=float(np.median(t)), ms_p95=float(np.percentile(t, 95)),
                iters=iters, include_preprocessing=include_preprocessing)


def train_step_memory(model, size, batch, device):
    if device.type != "cuda":
        return None
    model.train()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, 3, size, size, device=device)
    y = torch.arange(batch, device=device) % 5
    out = model(x)
    (torch.nn.functional.cross_entropy(out["logits"], y) + out["z"].sum() * 0 + out["alpha"].sum() * 0).backward()
    model.zero_grad(set_to_none=True)
    return torch.cuda.max_memory_allocated() / 2**30
