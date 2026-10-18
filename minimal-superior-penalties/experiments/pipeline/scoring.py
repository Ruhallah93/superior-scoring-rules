"""Offline scoring: the S_lambda family, baselines, and outcome metrics --
all computed from cached (probabilities, labels) arrays, never from a live
model. Matches `experiments/config.yaml`'s `conventions`, `lambda_grid`,
`baselines`, and `outcomes` sections.

Tie convention: single-valued argmax (`numpy.argmax`, lowest index on an
exact tie), matching config.yaml's `tie_convention: C`.
"""
import numpy as np

EPS = 1e-12


def brier_sum_per_sample(probs, labels):
    """Sum-form Brier score per sample. probs: (n, c), labels: (n,)."""
    n, c = probs.shape
    y = np.zeros((n, c))
    y[np.arange(n), labels] = 1.0
    return np.sum((probs - y) ** 2, axis=1)


def logloss_per_sample(probs, labels):
    n = len(labels)
    p_true = np.clip(probs[np.arange(n), labels], EPS, 1.0)
    return -np.log(p_true)


def penalty_indicator(probs, labels):
    """1[argmax(p) != true class], single-valued (lowest-index) argmax."""
    pred = np.argmax(probs, axis=1)
    return (pred != labels).astype(float)


def s_lambda_score(probs, labels, lam, base="brier"):
    """Mean S_lambda over the sample -- the aggregate selection criterion."""
    base_fn = brier_sum_per_sample if base == "brier" else logloss_per_sample
    per_sample = base_fn(probs, labels) + lam * penalty_indicator(probs, labels)
    return float(np.mean(per_sample))


def macro_f1(probs, labels, classes):
    pred = np.argmax(probs, axis=1)
    f1s = []
    for c in range(classes):
        tp = np.sum((pred == c) & (labels == c))
        fp = np.sum((pred == c) & (labels != c))
        fn = np.sum((pred != c) & (labels == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def error_rate(probs, labels):
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred != labels))


def nll(probs, labels):
    return float(np.mean(logloss_per_sample(probs, labels)))


def ece(probs, labels, bins=15, binning="equal_mass"):
    """Expected calibration error, equal-mass binning per config.yaml."""
    conf = probs.max(axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(float)
    order = np.argsort(conf)
    conf_s, correct_s = conf[order], correct[order]
    n = len(conf_s)
    edges = np.linspace(0, n, bins + 1).astype(int)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        bin_conf = conf_s[lo:hi].mean()
        bin_acc = correct_s[lo:hi].mean()
        total += (hi - lo) / n * abs(bin_conf - bin_acc)
    return float(total)


def weighted_mixture_score(probs, labels, weight):
    """Baseline R8: explicit mixture of (mean) Brier and error rate. Intended
    as a no-superiority control, but on inspection it is algebraically a
    strictly positive scalar multiple of a member of S_lambda itself for
    every weight tested except w=0, so it does not isolate the two effects
    (see config.yaml's `baselines` section).
    """
    n, c = probs.shape
    y = np.zeros((n, c))
    y[np.arange(n), labels] = 1.0
    mean_brier = float(np.mean(np.sum((probs - y) ** 2, axis=1)) / c)
    err = error_rate(probs, labels)
    return (1 - weight) * mean_brier + weight * err


def diagnostics(probs, labels):
    """margin_distribution / near_tie_rate / penalty_fire_rate, per
    config.yaml's `diagnostics` section.
    """
    sorted_probs = np.sort(probs, axis=1)
    top1, top2 = sorted_probs[:, -1], sorted_probs[:, -2]
    margin = top1 - top2
    return {
        "margin_mean": float(margin.mean()),
        "margin_median": float(np.median(margin)),
        "near_tie_rate": float(np.mean(margin < 0.01)),
        "penalty_fire_rate": float(np.mean(penalty_indicator(probs, labels))),
    }
