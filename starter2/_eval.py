"""Honest evaluation harness — exact score.py delay metric.

Modes:
  - within-language GroupKFold CV (grouped by turn_id)
  - cross-lingual transfer (train EN -> test HI and vice versa)

The hidden test set is "mostly Hindi", so train-EN->test-HI is our single
most important honest proxy for the real grade.
"""
import numpy as np

TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


def score_delay(y, durs, groups, p, budget=0.05):
    """Exact replica of score.py: best mean delay at <= budget cutoff rate."""
    turn_ids = list(set(groups))
    n_turns = len(turn_ids)
    y = np.asarray(y); durs = np.asarray(durs); p = np.asarray(p)
    groups = np.asarray(groups)
    is_hold = y == 0
    is_eot = y == 1
    best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0, "delay": TIMEOUT_S}
    for t in THRESHOLDS:
        fires = p >= t
        # holds that would fire (dur>delay handled per delay below)
        hold_fire_dur = durs[is_hold & fires]
        hold_fire_grp = groups[is_hold & fires]
        eot_fire = fires[is_eot]
        for d in DELAYS:
            cut_turns = set(hold_fire_grp[hold_fire_dur > d])
            cutoff = len(cut_turns) / max(1, n_turns)
            if cutoff > budget:
                continue
            lat = np.where(eot_fire, d, TIMEOUT_S).mean()
            if lat < best["latency"]:
                best = {"latency": float(lat), "cutoff": cutoff,
                        "threshold": float(t), "delay": float(d)}
    return best


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(p)
    ranks = np.empty(len(p), float); ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
