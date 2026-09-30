"""End-to-end smoke test on synthetic images (CPU, ~2-4 min). Run: python tests/smoke_test.py

Checks: split -> preprocess cache -> train (warm-up + joint stage) -> evaluate,
cached vs on-the-fly preprocessing equality, determinism across worker counts,
and that mono_mode=batch sends gradient into the encoder while mono_mode=ema does not.
"""
import csv, json, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eviproto_dr.losses import objective
from eviproto_dr.model import EviProtoDR
from eviproto_dr.preprocess import ben_graham, read_rgb


def run(*args):
    subprocess.run([sys.executable, "-m", "eviproto_dr.run", *map(str, args)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


def main():
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        g = i % 5
        im = np.zeros((300, 400, 3), np.uint8)
        cv2.circle(im, (200, 150), 140, (40 + 30 * g, 60, 90), -1)
        im = np.clip(im + rng.normal(0, 10, im.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(str(tmp / f"img{i}.jpg"), im)
        rows.append(dict(path=f"img{i}.jpg", grade=g))
    with open(tmp / "all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "grade"]); w.writeheader(); w.writerows(rows)

    run("split", "--source-csv", tmp / "all.csv", "--out", tmp / "split")
    run("preprocess", "--csv", tmp / "split/train.csv", tmp / "split/val.csv", tmp / "split/test.csv",
        "--out", tmp / "cache", "--size", 64, "--workers", 2)

    # cached PNG == on-the-fly preprocessing
    with open(tmp / "cache/train.csv") as f:
        r = next(csv.DictReader(f))
    assert np.array_equal(read_rgb(r["path"]), ben_graham(read_rgb(r["source"]), 64)), "cache mismatch"

    common = ["--train-csv", tmp / "cache/train.csv", "--val-csv", tmp / "cache/val.csv", "--epochs", 2,
              "--warmup", 1, "--size", 64, "--batch", 20, "--no-pretrained"]
    run("train", *common, "--out", tmp / "a", "--workers", 0)
    run("train", *common, "--out", tmp / "b", "--workers", 2)
    sa = torch.load(tmp / "a/model_weights.pt", weights_only=True)["model"]
    sb = torch.load(tmp / "b/model_weights.pt", weights_only=True)["model"]
    same = all(torch.equal(sa[k], sb[k]) for k in sa)
    print("identical weights with 0 vs 2 workers:", same)
    assert same

    run("evaluate", "--csv", tmp / "cache/test.csv", "--checkpoint", tmp / "a/model_weights.pt",
        "--size", 64, "--workers", 0, "--metrics-out", tmp / "a/test.json", "--predictions", tmp / "a/pred.csv")
    res = json.load(open(tmp / "a/test.json"))
    assert res["n"] == 9 and len(res["reliability_bins"]) == 10
    print("test metrics:", {k: res[k] for k in ("qwk", "ece", "auroc_uncertainty")})

    # gradient path of L_mono
    m = EviProtoDR(pretrained=False)
    m.prototypes.copy_(torch.nn.functional.normalize(torch.randn(5, 256), dim=1)); m.initialized.fill_(True)
    x, y = torch.randn(10, 3, 64, 64), torch.arange(10) % 5
    grads = {}
    for mode in ("ema", "batch"):
        m.zero_grad()
        o = m(x)
        _, t = objective(o, y, m.prototypes, epoch=20, mono_mode=mode)
        if t["mono"].requires_grad:
            raise AssertionError("terms must be detached")
        from eviproto_dr.losses import ordinal_monotonicity, batch_prototypes
        mono = ordinal_monotonicity(m.prototypes if mode == "ema" else batch_prototypes(o["z"], y), margin=2.0)
        grads[mode] = mono.requires_grad
    print("L_mono has gradient:", grads)
    assert grads == {"ema": False, "batch": True}
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
