import torch
from torch.nn import functional as F

K = 5


def dirichlet_kl(alpha):
    """KL[Dir(alpha) || Dir(1)] per sample."""
    s = alpha.sum(1)
    return (torch.lgamma(s) - torch.lgamma(alpha).sum(1) - torch.lgamma(alpha.new_tensor(float(K)))
            + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(s[:, None]))).sum(1))


def ordinal_monotonicity(p, margin=0.1):
    """Eq. (6). With 5 grades only k = 0, 1, 2 have a k+2 partner (3 hinge terms)."""
    dist = torch.cdist(p, p)
    return sum(F.relu(dist[k, k + 1] - dist[k, k + 2] + margin) for k in range(p.shape[0] - 2))


def batch_prototypes(z, y):
    """Differentiable per-grade batch means of z (all grades are present thanks to the sampler)."""
    onehot = F.one_hot(y, K).to(z.dtype)
    counts = onehot.sum(0)
    if not bool((counts > 0).all()):
        return None
    return F.normalize(onehot.T @ z / counts[:, None], dim=1)


def objective(out, y, prototypes, epoch, warmup=10, weights=(0.5, 0.1, 1.0, 0.5),
              tau=0.07, margin=0.1, mono_mode="ema"):
    """Eq. (15). Returns (total_loss, dict of detached tensors; no host sync per step).

    mono_mode="ema": L_mono on the EMA prototype buffers, as written in the
        manuscript. Buffers carry no gradient, so this term is logged but cannot
        change any weight.
    mono_mode="batch": L_mono on differentiable batch class means, which does
        back-propagate into the projection head and backbone.
    """
    l1, l2, l3, l4 = weights
    ce = F.cross_entropy(out["logits"], y)
    if epoch <= warmup:
        return ce, {"ce": ce.detach()}
    z, alpha, d, u = out["z"], out["alpha"], out["d"], out["u"]
    proto = F.cross_entropy(z @ prototypes.T / tau, y)
    p_mono = batch_prototypes(z, y) if mono_mode == "batch" else None
    mono = ordinal_monotonicity(prototypes if p_mono is None else p_mono, margin)
    s = alpha.sum(1)
    evid = (torch.log(s) - torch.log(alpha.gather(1, y[:, None]).squeeze(1))).mean()
    adjusted = 1.0 + (alpha - 1.0) * (1 - F.one_hot(y, K).float())
    kl = dirichlet_kl(adjusted).mean()
    proximity = 1 - d.gather(1, y[:, None]).squeeze(1) / d.max(1).values.clamp_min(1e-6)
    pucl = ((proximity - (1 - u)) ** 2).mean()
    lam_t = min(1.0, epoch / 10.0)  # manuscript schedule; equals 1 for every epoch > 10
    total = ce + l1 * proto + l2 * mono + l3 * evid + l4 * pucl + lam_t * kl
    terms = dict(ce=ce, proto=proto, mono=mono, evid=evid, pucl=pucl, kl=kl)
    return total, {k: v.detach() for k, v in terms.items()}
