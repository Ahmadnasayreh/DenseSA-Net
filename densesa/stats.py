"""Statistics for comparing models over repeated cross-validation."""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation for the Wilcoxon test."""
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    pos, neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((pos - neg) / (pos + neg))


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / diff.size)


def effect_label(delta):
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def bootstrap_ci(a, b, n_boot=10000, alpha=0.05, seed=42):
    """Percentile bootstrap CI of the mean paired difference a - b."""
    d = np.asarray(a) - np.asarray(b)
    if d.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def _clean(values):
    # Two models with the same number of correct images in a fold should tie,
    # but accuracies computed through different code paths can differ in the
    # last bit. Rounding makes those ties exact.
    return np.round(np.asarray(values, dtype=float), 10)


def friedman(scores):
    """scores: dict model -> list of fold accuracies (same folds for all)."""
    scores = {k: _clean(v) for k, v in scores.items()}
    names = list(scores)
    if len(names) < 3:
        return None
    n = len(scores[names[0]])
    chi2, p = friedmanchisquare(*[scores[m] for m in names])
    return {"k_models": len(names), "n_observations": n,
            "Friedman_chi2": float(chi2), "p_value": float(p),
            "theoretical_max_chi2": n * (len(names) - 1),
            "min_possible_wilcoxon_p": 2 / 2 ** n}


def pairwise(scores, proposed="DenseSANet", collapse_below=0.5):
    """Wilcoxon signed-rank tests of the proposed model against each baseline.

    A baseline fold below collapse_below is a training divergence. Those
    comparisons are reported twice: with all folds and without the diverged ones.
    """
    rows = []
    scores = {k: _clean(v) for k, v in scores.items()}
    a_all = scores[proposed]
    for other, b_all in scores.items():
        if other == proposed:
            continue
        subsets = [("all observations", np.ones_like(b_all, dtype=bool))]
        diverged = b_all < collapse_below
        if diverged.any() and (~diverged).sum() >= 3:
            subsets.append((f"excluding {diverged.sum()} diverged", ~diverged))
        for label, keep in subsets:
            a, b = a_all[keep], b_all[keep]
            try:
                # normal approximation (Z statistic), zero differences dropped
                w, p = wilcoxon(a, b, zero_method="wilcox", method="approx")
            except ValueError:
                w, p = float("nan"), float("nan")
            delta = cliffs_delta(a, b)
            mean_d, lo, hi = bootstrap_ci(a, b)
            rows.append({"Comparison": f"{proposed} vs {other}", "Subset": label,
                         "n": int(keep.sum()), "Wilcoxon_W": float(w), "p_value": float(p),
                         "Rank_biserial_r": rank_biserial(a, b), "Cliffs_delta": delta,
                         "Effect": effect_label(delta), "Mean_diff": mean_d,
                         "CI95_low": lo, "CI95_high": hi,
                         "Folds_won": f"{int((a > b).sum())}/{a.size}"})
    return pd.DataFrame(rows)
