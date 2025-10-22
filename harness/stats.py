"""Statistical procedures used by the analysis (pre-specified in ANALYSIS_PLAN.md).

Everything here is deterministic given the seed. Dependencies: numpy, scipy.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------- proportions
def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def newcombe_diff(k1, n1, k2, n2):
    """Newcombe hybrid-score 95% CI for p1 - p2 (independent samples)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"), float("nan"))
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (d, lo, hi)


def mcnemar_exact(b: int, c: int):
    """Exact McNemar test on discordant pairs (b: only method 1 positive, c: only method 2)."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    p = min(1.0, 2 * sps.binom.cdf(min(b, c), n, 0.5))
    return {"b": b, "c": c, "p": float(p)}


def paired_diff_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, B: int = 4000):
    """Bootstrap 95% CI for mean(x) - mean(y) over paired binary outcomes."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    d = x - y
    idx = rng.integers(0, n, size=(B, n))
    boot = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        prev = max(prev, a)
        adj[k] = prev
    return adj


# ---------------------------------------------------------------- cluster bootstrap for ratios of means
def bootstrap_ratio(per_class_numden_tel, per_class_numden_per, rng, B=4000):
    """per_class_numden_*: list over classes of (array of session-level 0/1 outcomes).
    Ratio of class-averaged rates, telemetry / perimeter; sessions resampled within class."""
    def stat(samples_t, samples_p):
        mt = np.mean([s.mean() if len(s) else np.nan for s in samples_t])
        mp = np.mean([s.mean() if len(s) else np.nan for s in samples_p])
        return mt / mp if mp > 0 else np.nan
    point = stat(per_class_numden_tel, per_class_numden_per)
    boots = []
    for _ in range(B):
        st = [s[rng.integers(0, len(s), len(s))] if len(s) else s for s in per_class_numden_tel]
        sp = [s[rng.integers(0, len(s), len(s))] if len(s) else s for s in per_class_numden_per]
        boots.append(stat(st, sp))
    boots = np.array([b for b in boots if b == b and np.isfinite(b)])
    if len(boots) == 0:
        return (point, float("nan"), float("nan"))
    return (float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


# ---------------------------------------------------------------- survival (time to detection)
def kaplan_meier(times: np.ndarray, events: np.ndarray, horizon: int):
    """KM estimate of P(detected by t) = 1 - S(t) on an integer grid 0..horizon,
    with Greenwood pointwise 95% CIs (log-log transform). times are actions
    elapsed; events=1 detected, 0 censored at session end."""
    times, events = np.asarray(times, float), np.asarray(events, int)
    grid = np.arange(0, horizon + 1)
    S, var_sum, surv, lo, hi = 1.0, 0.0, [], [], []
    uniq = np.unique(times[events == 1])
    j = 0
    s_at, v_at = {}, {}
    for t in uniq:
        at_risk = np.sum(times >= t)
        d = np.sum((times == t) & (events == 1))
        if at_risk > 0:
            S *= (1 - d / at_risk)
            if at_risk > d:
                var_sum += d / (at_risk * (at_risk - d))
        s_at[t] = S; v_at[t] = var_sum
    S, V = 1.0, 0.0
    for g in grid:
        for t in uniq:
            if t <= g:
                S, V = s_at[t], v_at[t]
        surv.append(S)
        if 0 < S < 1 and V > 0:
            se = math.sqrt(V) / abs(math.log(S))
            l = S ** math.exp(1.959964 * se); h = S ** math.exp(-1.959964 * se)
            lo.append(min(l, h)); hi.append(max(l, h))
        else:
            lo.append(S); hi.append(S)
    det = [1 - s for s in surv]
    return {"grid": grid.tolist(), "detected": det, "lo": [1 - h for h in hi], "hi": [1 - l for l in lo],
            "n": int(len(times)), "n_events": int(events.sum()),
            "median": _km_median(grid, det)}


def _km_median(grid, det):
    for g, d in zip(grid, det):
        if d >= 0.5:
            return int(g)
    return None


def logrank(t1, e1, t2, e2):
    """Two-sample log-rank test. Returns chi2 statistic and p-value."""
    t1, e1, t2, e2 = map(np.asarray, (t1, e1, t2, e2))
    times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]])) if (e1.sum() + e2.sum()) else np.array([])
    O1 = E1 = V = 0.0
    for t in times:
        n1, n2 = np.sum(t1 >= t), np.sum(t2 >= t)
        d1, d2 = np.sum((t1 == t) & (e1 == 1)), np.sum((t2 == t) & (e2 == 1))
        n, d = n1 + n2, d1 + d2
        if n == 0:
            continue
        E1 += d * n1 / n
        O1 += d1
        if n > 1:
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if V <= 0:
        return {"chi2": 0.0, "p": 1.0}
    chi2 = (O1 - E1) ** 2 / V
    return {"chi2": float(chi2), "p": float(sps.chi2.sf(chi2, 1))}


# ---------------------------------------------------------------- ROC / AUC
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = sps.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(scores, labels, rng, B=2000):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    point = auc(scores, labels)
    if point != point:
        return (point, float("nan"), float("nan"))
    pos_i, neg_i = np.where(labels == 1)[0], np.where(labels == 0)[0]
    boots = []
    for _ in range(B):
        p = pos_i[rng.integers(0, len(pos_i), len(pos_i))]
        n = neg_i[rng.integers(0, len(neg_i), len(neg_i))]
        boots.append(auc(np.concatenate([scores[p], scores[n]]), np.concatenate([np.ones(len(p)), np.zeros(len(n))])))
    return (point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def roc_curve(scores, labels, n_points=50):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    ths = np.unique(np.concatenate([[np.inf], np.quantile(scores, np.linspace(0, 1, n_points)), [-np.inf]]))[::-1]
    fpr, tpr = [], []
    P, N = max(1, labels.sum()), max(1, (1 - labels).sum())
    for t in ths:
        pred = scores >= t
        tpr.append(float(np.sum(pred & (labels == 1)) / P))
        fpr.append(float(np.sum(pred & (labels == 0)) / N))
    return {"fpr": fpr, "tpr": tpr}


# ---------------------------------------------------------------- logistic combiner (IRLS) with K-fold CV
def logistic_fit(X, y, l2=1e-2, iters=50):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        g = X.T @ (y - p) - l2 * w
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def logistic_predict(w, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-X @ w))


def cv_combiner_auc(X, y, rng, k=5):
    X, y = np.asarray(X, float), np.asarray(y, int)
    n = len(y)
    if n < 20 or y.sum() < 5 or (1 - y).sum() < 5:
        return {"auc": float("nan"), "coef": None}
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    preds = np.zeros(n)
    for f in folds:
        train = np.setdiff1d(order, f)
        w = logistic_fit(X[train], y[train])
        preds[f] = logistic_predict(w, X[f])
    w_all = logistic_fit(X, y)
    return {"auc": auc(preds, y), "auc_ci": auc_ci(preds, y, rng, B=1000), "coef": w_all.tolist()}


# ---------------------------------------------------------------- conformal threshold
def conformal_threshold(cal_scores, alpha: float) -> float:
    """Split-conformal (1-alpha) quantile: with n exchangeable calibration
    scores, P(new score > threshold) <= alpha (marginally)."""
    s = np.sort(np.asarray(cal_scores, float))
    n = len(s)
    if n == 0:
        return 1.0
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])
