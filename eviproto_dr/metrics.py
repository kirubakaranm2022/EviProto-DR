import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

K = 5


def reliability_bins(y, probs, n_bins=10):
    """Per-bin data for reliability diagrams (equal-width confidence bins)."""
    pred, conf = probs.argmax(1), probs.max(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for i in range(n_bins):
        m = bins == i
        out.append(dict(lower=i / n_bins, upper=(i + 1) / n_bins, count=int(m.sum()),
                        accuracy=float((pred[m] == y[m]).mean()) if m.any() else None,
                        confidence=float(conf[m].mean()) if m.any() else None))
    return out


def ece_from_bins(bins, n):
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"]))


def referral_sweep(y, pred, u, thresholds=np.round(np.arange(0.10, 0.61, 0.05), 2)):
    rows = []
    for t in thresholds:
        keep = u <= t
        rows.append(dict(threshold=float(t), referral_rate=float((~keep).mean()),
                         non_deferred_accuracy=float((pred[keep] == y[keep]).mean()) if keep.any() else None,
                         referred_error_share=float(((pred != y) & ~keep).sum() / max((pred != y).sum(), 1))))
    return rows


def compute(y, probs, uncertainty, threshold=0.3, distances=None, detail=False):
    y, probs, u = np.asarray(y), np.asarray(probs), np.asarray(uncertainty)
    pred = probs.argmax(1)
    bins = reliability_bins(y, probs)
    wrong = (pred != y).astype(int)
    keep = u <= threshold
    res = dict(
        n=int(len(y)),
        accuracy=float(accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0)),
        qwk=float(cohen_kappa_score(y, pred, weights="quadratic", labels=list(range(K)))),
        mae=float(np.abs(y - pred).mean()),
        large_error_rate=float((np.abs(y - pred) >= 2).mean()),
        ece=ece_from_bins(bins, len(y)),
        brier=float(np.square(probs - np.eye(K)[y]).sum(1).mean()),
        auroc_uncertainty=float(roc_auc_score(wrong, u)) if len(set(wrong)) == 2 else None,
        referral_rate=float((~keep).mean()),
        non_deferred_accuracy=float((pred[keep] == y[keep]).mean()) if keep.any() else None,
        per_class_f1=[float(v) for v in f1_score(y, pred, average=None, labels=list(range(K)), zero_division=0)],
    )
    if distances is not None:
        res["mean_intra_grade_distance"] = float(np.asarray(distances)[np.arange(len(y)), y].mean())
    if detail:
        res["reliability_bins"] = bins
        res["referral_sweep"] = referral_sweep(y, pred, u)
    return res


def prototype_geometry(prototypes):
    """OMS and Proto-EDC from a (5, D) prototype matrix.

    OMS: fraction of k in {0,1,2} with d(k,k+1) < d(k,k+2). Three checks exist
    for five grades, so OMS takes values in {0, 1/3, 2/3, 1}.
    Proto-EDC: correlation between the 10 pairwise prototype distances and |k-l|;
    both Pearson and Spearman are returned so the reported variant is explicit.
    """
    p = np.asarray(prototypes, dtype=np.float64)
    dist = np.linalg.norm(p[:, None] - p[None], axis=-1)
    checks = [bool(dist[k, k + 1] < dist[k, k + 2]) for k in range(len(p) - 2)]
    iu = np.triu_indices(len(p), 1)
    gaps = np.abs(iu[0] - iu[1])
    return dict(oms=float(np.mean(checks)), oms_checks=checks,
                proto_edc_pearson=float(stats.pearsonr(dist[iu], gaps)[0]),
                proto_edc_spearman=float(stats.spearmanr(dist[iu], gaps)[0]),
                distance_matrix=dist.round(6).tolist())
