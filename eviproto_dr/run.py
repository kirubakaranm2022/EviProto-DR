import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from . import __version__
from .data import BalancedBatchSampler, FundusDataset
from .losses import objective
from .metrics import compute, prototype_geometry
from .model import EviProtoDR
from .preprocess import cache_csv, resolve


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


# ---------------------------------------------------------------- split
def make_split(args):
    with open(args.source_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"path", "grade"} <= rows[0].keys():
        raise ValueError("Source CSV needs columns path,grade")
    for r in rows:
        r["path"] = str(resolve(args.source_csv, r["path"]).resolve())
    dropped = [r for r in rows if int(r["grade"]) not in range(5)]
    if dropped:
        print(f"excluding {len(dropped)} rows with grades outside 0..4 (e.g. DDR ungradable)")
        rows = [r for r in rows if int(r["grade"]) in range(5)]
    labels = np.array([int(r["grade"]) for r in rows])
    groups = np.array([r[args.group_column] for r in rows]) if args.group_column else None
    idx = np.arange(len(rows))
    if groups is None:
        tr, rest = train_test_split(idx, test_size=.30, stratify=labels, random_state=args.seed)
        va, te = train_test_split(rest, test_size=.50, stratify=labels[rest], random_state=args.seed)
    else:  # patient-level split: no patient appears in two subsets
        from sklearn.model_selection import StratifiedGroupKFold
        sgk = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=args.seed)
        folds = [f for _, f in sgk.split(idx, labels, groups)]
        te, va, tr = np.concatenate(folds[:3]), np.concatenate(folds[3:6]), np.concatenate(folds[6:])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, ids in [("train", tr), ("val", va), ("test", te)]:
        p = out / f"{name}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows[i] for i in sorted(ids))
        manifest[name] = dict(n=int(len(ids)), grades=np.bincount(labels[ids], minlength=5).tolist(), sha256=sha256(p))
        print(name, manifest[name])
    json.dump(dict(seed=args.seed, source=str(args.source_csv), splits=manifest), open(out / "manifest.json", "w"), indent=2)


# ---------------------------------------------------------------- data
def loader(csv_path, batch, workers, size, train=False, seed=42):
    ds = FundusDataset(csv_path, train=train, size=size)
    # A private generator keeps DataLoader seed draws out of the global RNG, which
    # drives dropout / stochastic depth; otherwise results depend on --workers.
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
              generator=torch.Generator().manual_seed(seed))
    if workers > 0:
        kw["prefetch_factor"] = 4
    if train:
        return DataLoader(ds, batch_sampler=BalancedBatchSampler(ds.labels, batch, seed=seed), **kw)
    return DataLoader(ds, batch_size=batch, shuffle=False, **kw)


@torch.no_grad()
def evaluate(model, dl, device, threshold=.3, export=None, embeddings=None, amp=False, detail=False):
    model.eval()
    ys, ps, us, ds, zs, paths = [], [], [], [], [], []
    for x, y, p in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            o = model(x)
        ys.append(y.numpy()); ps.append(o["prob"].cpu().numpy()); us.append(o["u"].cpu().numpy())
        ds.append(o["d"].cpu().numpy()); paths.extend(p)
        if embeddings:
            zs.append(o["z"].cpu().numpy())
    y, prob, u, d = map(np.concatenate, (ys, ps, us, ds))
    if export:
        with open(export, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "grade", "prediction", "uncertainty", "referred"]
                       + [f"p{k}" for k in range(5)] + [f"d{k}" for k in range(5)])
            for i in range(len(y)):
                w.writerow([paths[i], int(y[i]), int(prob[i].argmax()), float(u[i]), bool(u[i] > threshold),
                            *map(float, prob[i]), *map(float, d[i])])
    if embeddings:
        np.savez_compressed(embeddings, z=np.concatenate(zs), y=y, u=u,
                            prototypes=model.prototypes.cpu().numpy())
    return compute(y, prob, u, threshold, distances=d, detail=detail)


# ---------------------------------------------------------------- train
def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    print("device", device, torch.cuda.get_device_name() if device.type == "cuda" else "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = loader(args.train_csv, args.batch, args.workers, args.size, True, args.seed)
    val = loader(args.val_csv, args.batch, args.workers, args.size)
    model = EviProtoDR(pretrained=not args.no_pretrained, beta=args.beta).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - args.warmup)
    start = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "rng" in state:
            set_rng_state(state["rng"])
        start = state["epoch"] + 1
    config = dict(vars(args), version=__version__,
                  train_sha256=sha256(args.train_csv), val_sha256=sha256(args.val_csv))
    json.dump(config, open(out / "run_config.json", "w"), indent=2, default=str)
    if args.mono_mode == "ema":
        print("note: mono_mode=ema -> L_mono is logged but has no gradient (prototypes are EMA buffers)")
    log = open(out / "train_log.jsonl", "a")
    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr.batch_sampler.set_epoch(epoch - 1)
        sums = torch.zeros(5, model.prototypes.shape[1], device=device)
        counts = torch.zeros(5, device=device)
        running, n_steps = None, 0
        for x, y, _ in tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                o = model(x)
            loss, terms = objective(o, y, model.prototypes, epoch, warmup=args.warmup, mono_mode=args.mono_mode)
            loss.backward()
            opt.step()
            if epoch == 1:
                sums.index_add_(0, y, o["z"].detach())
                counts += torch.bincount(y, minlength=5)
            else:
                model.update_prototypes(o["z"], y)
            step = torch.stack([loss.detach()] + list(terms.values()))
            running = step if running is None else running + step
            n_steps += 1
        if epoch == 1:
            model.initialize_from_means(sums, counts)
        if epoch > args.warmup:
            scheduler.step()
        avg = (running / n_steps).tolist()
        rec = dict(epoch=epoch, loss=avg[0], terms=dict(zip(terms.keys(), avg[1:])),
                   lr=opt.param_groups[0]["lr"], minutes=(time.time() - t0) / 60,
                   geometry={k: v for k, v in prototype_geometry(model.prototypes.cpu()).items()
                             if k != "distance_matrix"})
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            rec["validation"] = evaluate(model, val, device, args.threshold, amp=amp)
        if device.type == "cuda":
            rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                        scheduler=scheduler.state_dict(), seed=args.seed, threshold=args.threshold,
                        rng=rng_state(), config=config), out / "last.pt")
        if args.save_prototypes:
            np.save(out / f"prototypes_epoch{epoch:03d}.npy", model.prototypes.cpu().numpy())
        if args.stop_after and epoch >= args.stop_after and epoch < args.epochs:
            print(f"stopped after epoch {epoch}; continue with --resume {out / 'last.pt'}")
            return
    # weights-only file for sharing (loads with torch.load(weights_only=True))
    torch.save(dict(model=model.state_dict(), seed=args.seed, threshold=args.threshold,
                    size=args.size, version=__version__), out / "model_weights.pt")
    print("final checkpoint:", out / "last.pt")


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=args.checkpoint.endswith("model_weights.pt"))
    size = state.get("size") or state.get("config", {}).get("size", args.size)
    model = EviProtoDR(pretrained=False).to(device)
    model.load_state_dict(state["model"])
    dl = loader(args.csv, args.batch, args.workers, size)
    result = evaluate(model, dl, device, args.threshold, args.predictions, args.embeddings, detail=True)
    result.update(seed=state.get("seed"), checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                  csv_sha256=sha256(args.csv), geometry=prototype_geometry(model.prototypes.cpu()))
    print(json.dumps({k: v for k, v in result.items() if k not in ("reliability_bins", "referral_sweep")}, indent=2))
    if args.metrics_out:
        json.dump(result, open(args.metrics_out, "w"), indent=2)


def preprocess(args):
    for c in args.csv:
        print("cached ->", cache_csv(c, args.out, args.size, args.workers))


def profile(args):
    from .profile import count_params, environment, flops, latency, train_step_memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EviProtoDR(pretrained=False).to(device)
    res = dict(environment=environment(), params=count_params(model), flops=flops(model, args.size, device),
               latency=latency(model, args.size, device, args.iters),
               latency_with_preprocessing=latency(model, args.size, device, args.iters, include_preprocessing=True),
               train_peak_memory_gb=train_step_memory(model, args.size, args.batch, device))
    print(json.dumps(res, indent=2))


def aggregate(args):
    from .stats import compare
    res = compare(args.a, args.b, args.metrics)
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


def main():
    p = argparse.ArgumentParser(prog="python -m eviproto_dr.run")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split", help="grade-stratified 70/15/15 split")
    s.add_argument("--source-csv", required=True); s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--group-column", help="e.g. patient_id, for a patient-level split")
    c = sub.add_parser("preprocess", help="cache Ben Graham images once")
    c.add_argument("--csv", nargs="+", required=True); c.add_argument("--out", required=True)
    c.add_argument("--size", type=int, default=512); c.add_argument("--workers", type=int, default=8)
    t = sub.add_parser("train")
    t.add_argument("--train-csv", required=True); t.add_argument("--val-csv", required=True)
    t.add_argument("--out", required=True); t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--warmup", type=int, default=10); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--beta", type=float, default=0.99)
    t.add_argument("--mono-mode", choices=["ema", "batch"], default="ema")
    t.add_argument("--resume"); t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--eval-every", type=int, default=1)
    t.add_argument("--stop-after", type=int, help="end this session after N epochs (for time-limited cloud GPUs)")
    t.add_argument("--save-prototypes", action="store_true", help="save prototypes each epoch (Figure 6)")
    t.add_argument("--amp", action="store_true", help="bf16 autocast for the backbone (faster; not the reported setting)")
    t.add_argument("--channels-last", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--csv", required=True); e.add_argument("--checkpoint", required=True)
    e.add_argument("--predictions"); e.add_argument("--metrics-out"); e.add_argument("--embeddings")
    f = sub.add_parser("profile")
    f.add_argument("--iters", type=int, default=200)
    a = sub.add_parser("aggregate", help="mean/SD/CI + paired tests across seeds")
    a.add_argument("--a", required=True, help='glob, e.g. "runs/eviproto_s*/test_metrics.json"')
    a.add_argument("--b", required=True, help="glob for the comparison configuration")
    a.add_argument("--metrics", nargs="+", default=["qwk", "accuracy", "macro_f1", "ece", "brier", "auroc_uncertainty"])
    a.add_argument("--out")
    for q in (t, e, f):
        q.add_argument("--batch", type=int, default=32); q.add_argument("--size", type=int, default=512)
    for q in (t, e):
        q.add_argument("--workers", type=int, default=4); q.add_argument("--threshold", type=float, default=.3)
    args = p.parse_args()
    dict(split=make_split, preprocess=preprocess, train=train, evaluate=test,
         profile=profile, aggregate=aggregate)[args.command](args)


if __name__ == "__main__":
    main()
