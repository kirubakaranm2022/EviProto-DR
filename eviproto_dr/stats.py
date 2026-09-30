"""Seed aggregation and paired significance tests (Section 4.2.2)."""
import glob
import json

import numpy as np
from scipy import stats


def load(pattern):
    runs = {}
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        if r.get("seed") is None:
            raise ValueError(f"{f} has no seed; re-run evaluate with a checkpoint that stores it")
        runs[int(r["seed"])] = r
    if not runs:
        raise FileNotFoundError(pattern)
    return runs


def summary(values):
    v = np.asarray(values, float)
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else 0.0
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
    return dict(n=n, mean=float(v.mean()), sd=float(sd), ci95=[float(v.mean() - half), float(v.mean() + half)])


def paired_test(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    n = len(diff)
    normal_p = float(stats.shapiro(diff).pvalue) if n >= 3 and np.ptp(diff) > 0 else None
    if normal_p is not None and normal_p < 0.05:
        test, p = "wilcoxon", float(stats.wilcoxon(diff).pvalue)
    else:
        test, p = "paired_t", float(stats.ttest_rel(a, b).pvalue)
    sd = diff.std(ddof=1)
    note = None
    if test == "wilcoxon" and n <= 5:
        note = "Exact two-sided Wilcoxon with n<=5 cannot give p < 0.0625"
    return dict(test=test, shapiro_p=normal_p, p=p, mean_diff=float(diff.mean()),
                cohens_dz=float(diff.mean() / sd) if sd > 0 else None, note=note)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def compare(pattern_a, pattern_b, metrics):
    a, b = load(pattern_a), load(pattern_b)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        raise ValueError(f"Need >= 2 shared seeds, found {seeds}")
    out = dict(seeds=seeds, metrics={})
    for m in metrics:
        va = [a[s][m] for s in seeds]
        vb = [b[s][m] for s in seeds]
        out["metrics"][m] = dict(a=summary(va), b=summary(vb), paired=paired_test(va, vb))
    adj = holm([out["metrics"][m]["paired"]["p"] for m in metrics])
    for m, p in zip(metrics, adj):
        out["metrics"][m]["paired"]["p_holm"] = p
    return out
