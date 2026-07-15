"""EOT training pipeline.

Models: LR + HistGBT + RandomForest → sigmoid calibration → weighted ensemble.
CV: GroupKFold by turn_id, scored using score.py's actual delay metric.
Final training: English + Hindi combined.

Usage:
    python train.py --data_dirs eot_handout/eot_data/english eot_handout/eot_data/hindi
"""
import argparse
import csv
import os
import pickle
import warnings
from itertools import product

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from features import load_wav, extract_features, FEATURE_NAMES, N_FEATURES

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dirs):
    """Load pauses from one or more data directories."""
    rows_all, X_all, y_all, groups_all, meta_all = [], [], [], [], []
    wav_cache = {}

    for data_dir in data_dirs:
        label_path = os.path.join(data_dir, "labels.csv")
        rows = list(csv.DictReader(open(label_path)))

        # Pre-build turn_pauses dict for cumulative discourse features (causal)
        turn_pauses = {}
        for r in rows:
            turn_pauses.setdefault(r["turn_id"], []).append(r)

        for r in rows:
            path = os.path.join(data_dir, r["audio_file"])
            if path not in wav_cache:
                wav_cache[path] = load_wav(path)
            x, sr = wav_cache[path]

            pause_index    = int(r["pause_index"])
            n_pauses       = pause_index + 1
            pause_start    = float(r["pause_start"])
            pause_end      = float(r["pause_end"])
            tid            = r["turn_id"]

            # Prior pause durations only (pause_index < current) — causal
            prev_durs = [
                float(prev["pause_end"]) - float(prev["pause_start"])
                for prev in turn_pauses[tid]
                if int(prev["pause_index"]) < pause_index
            ]

            feat = extract_features(x, sr, pause_start, pause_index, n_pauses,
                                    prev_pause_durs=prev_durs)
            label = 1 if r["label"] == "eot" else 0

            X_all.append(feat)
            y_all.append(label)
            groups_all.append(tid)
            rows_all.append(r)
            meta_all.append({
                "turn_id":    tid,
                "pause_index": pause_index,
                "pause_dur":  pause_end - pause_start,
                "label":      r["label"],
            })

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=int)
    return X, y, groups_all, meta_all


# ─────────────────────────────────────────────────────────────────────────────
# Scoring harness — replicates score.py logic exactly
# ─────────────────────────────────────────────────────────────────────────────

TIMEOUT_S   = 1.6
THRESHOLDS  = np.round(np.arange(0.05, 1.00, 0.05), 3)
DELAYS      = np.round(np.arange(0.10, 1.65, 0.05), 3)


def _score_predictions(meta, p_eot, budget=0.05):
    """
    Replicate score.py: sweep (threshold × delay), return best mean delay
    achievable at ≤ budget false-cutoff rate.
    """
    best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0, "delay": TIMEOUT_S}
    turn_ids = list({m["turn_id"] for m in meta})

    for t in THRESHOLDS:
        for d in DELAYS:
            turns_cut = set()
            latencies = []
            for m, p in zip(meta, p_eot):
                fires = p >= t
                if m["label"] == "hold":
                    if fires and d < m["pause_dur"]:
                        turns_cut.add(m["turn_id"])
                else:
                    latencies.append(d if fires else TIMEOUT_S)
            cutoff_rate = len(turns_cut) / max(1, len(turn_ids))
            mean_lat = float(np.mean(latencies)) if latencies else TIMEOUT_S
            if cutoff_rate <= budget and mean_lat < best["latency"]:
                best = {"latency": mean_lat, "cutoff": cutoff_rate,
                        "threshold": t, "delay": d}
    return best


def cv_delay_score(pipeline, X, y, groups, meta, n_splits=5, budget=0.05):
    """
    GroupKFold CV grouped by turn_id.
    Returns (mean_delay_ms, std_delay_ms, list_of_fold_results).
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_latencies = []
    fold_results   = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val      = X[val_idx]
        meta_val   = [meta[i] for i in val_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_tr, y_tr)

        p_val = pipeline.predict_proba(X_val)[:, 1]
        res   = _score_predictions(meta_val, p_val, budget)
        fold_latencies.append(res["latency"])
        fold_results.append(res)

    arr = np.array(fold_latencies)
    return float(arr.mean()), float(arr.std()), fold_results


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def make_lr():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
        ("clf",     LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def make_hgbt():
    base = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,
        min_samples_leaf=15,
        learning_rate=0.05,
        l2_regularization=0.1,
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", CalibratedClassifierCV(base, method="isotonic", cv=5)),
    ])


def make_rf():
    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
        ("clf",     CalibratedClassifierCV(base, method="isotonic", cv=5)),
    ])


def make_et():
    """ExtraTreesClassifier — starter found this competitive with GBT."""
    base = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
        ("clf",     CalibratedClassifierCV(base, method="isotonic", cv=5)),
    ])


def make_gbt():
    """Classic GradientBoostingClassifier — winner in the starter codebase."""
    base = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=4,
        random_state=42,
    )
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf",    CalibratedClassifierCV(base, method="isotonic", cv=5)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble helpers
# ─────────────────────────────────────────────────────────────────────────────

class WeightedEnsemble:
    """Predict-proba weighted average of pre-fitted models."""
    def __init__(self, models, weights):
        self.models  = models
        self.weights = np.array(weights) / sum(weights)

    def predict_proba(self, X):
        proba = np.zeros((len(X), 2))
        for m, w in zip(self.models, self.weights):
            proba += w * m.predict_proba(X)
        return proba


def _best_ensemble_weights(pipelines, X_val_list, meta_val_list, budget=0.05):
    """
    Grid-search ensemble weights over {0.0, 0.25, 0.5, 0.75, 1.0} for 3 models.
    Returns (best_weights, best_delay).
    """
    weight_choices = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_w, best_lat = None, TIMEOUT_S * 10

    for w0, w1, w2 in product(weight_choices, repeat=3):
        if w0 + w1 + w2 < 1e-6:
            continue
        weights = np.array([w0, w1, w2])
        weights = weights / weights.sum()
        # evaluate on each val fold
        fold_lats = []
        for p_list, m_list in zip(X_val_list, meta_val_list):
            # p_list is list of (p_proba_array) per model
            p_ens = sum(w * p for w, p in zip(weights, p_list))
            res = _score_predictions(m_list, p_ens, budget)
            fold_lats.append(res["latency"])
        mean_lat = float(np.mean(fold_lats))
        if mean_lat < best_lat:
            best_lat, best_w = mean_lat, weights.tolist()
    return best_w, best_lat


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True,
                    help="One or more data directories (english and/or hindi)")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model_out", default="model.pkl")
    ap.add_argument("--cv_splits", type=int, default=5)
    args = ap.parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    X, y, groups, meta = load_dataset(args.data_dirs)
    n_eot  = int(y.sum())
    n_hold = int((y == 0).sum())
    print(f"  {len(X)} pauses total — {n_eot} EOT, {n_hold} HOLD")
    print(f"  {N_FEATURES} features: {FEATURE_NAMES}")
    nan_counts = np.isnan(X).sum(axis=0)
    noisy = [(FEATURE_NAMES[i], int(nan_counts[i])) for i in range(N_FEATURES) if nan_counts[i] > 0]
    if noisy:
        print(f"  NaN features: {noisy}")

    # ── 2. CV evaluation of all 3 models ──────────────────────────────────────
    print("\n=== Cross-validation (GroupKFold, n_splits={}) ===".format(args.cv_splits))
    pipelines = {
        "LR":   make_lr(),
        "HGBT": make_hgbt(),
        "RF":   make_rf(),
        "ET":   make_et(),
        "GBT":  make_gbt(),
    }

    cv_results = {}
    for name, pipe in pipelines.items():
        print(f"\n  [{name}] evaluating...", end=" ", flush=True)
        mean_d, std_d, fold_res = cv_delay_score(
            pipe, X, y, groups, meta, n_splits=args.cv_splits
        )
        cv_results[name] = (mean_d, std_d, fold_res)
        print(f"delay = {mean_d*1000:.0f} ± {std_d*1000:.0f} ms")

    # ── 3. Ensemble weight search ─────────────────────────────────────────────
    print("\n=== Ensemble weight search ===")
    gkf = GroupKFold(n_splits=args.cv_splits)
    # Collect per-fold val probas for each model
    val_probas_per_fold = []   # list of folds → list of models → p array
    meta_val_per_fold  = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        fold_ps = []
        for pipe in pipelines.values():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_tr, y_tr)
            fold_ps.append(pipe.predict_proba(X_val)[:, 1])
        val_probas_per_fold.append(fold_ps)
        meta_val_per_fold.append([meta[i] for i in val_idx])

    best_weights, ens_delay = _best_ensemble_weights(
        list(pipelines.values()),
        val_probas_per_fold,
        meta_val_per_fold,
    )
    print(f"  Best weights (LR/HGBT/RF): {[f'{w:.2f}' for w in best_weights]}")
    print(f"  Ensemble CV delay: {ens_delay*1000:.0f} ms")
    cv_results["Ensemble"] = (ens_delay, 0.0, [])

    # ── 4. Model selection ────────────────────────────────────────────────────
    print("\n=== Model comparison ===")
    print(f"  {'Model':<12} {'CV delay (ms)':>14} {'± ms':>8}")
    print(f"  {'-'*36}")
    for name, (mean_d, std_d, _) in cv_results.items():
        print(f"  {name:<12} {mean_d*1000:>14.0f} {std_d*1000:>8.0f}")

    best_model_name = min(cv_results, key=lambda k: cv_results[k][0])
    print(f"\n  ► Selected: {best_model_name} (lowest CV delay)")

    # ── 5. Final training on ALL data ─────────────────────────────────────────
    print("\n=== Final training on ALL data ===")
    if best_model_name == "Ensemble":
        # Refit all on all data
        final_models = []
        for name, pipe in pipelines.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X, y)
            final_models.append(pipe)
        final_model = WeightedEnsemble(final_models, best_weights)
    else:
        final_pipeline = pipelines[best_model_name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_pipeline.fit(X, y)
        final_model = final_pipeline

    # Save
    artifact = {
        "model": final_model,
        "model_name": best_model_name,
        "cv_results": {k: (v[0], v[1]) for k, v in cv_results.items()},
        "feature_names": FEATURE_NAMES,
    }
    with open(args.model_out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  Saved -> {args.model_out}")

    # ── 6. Write predictions on the training data ──────────────────────────────
    p_all = final_model.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for m, p in zip(meta, p_all):
            w.writerow([m["turn_id"], m["pause_index"], f"{p:.4f}"])
    print(f"  Predictions -> {args.out} ({len(meta)} rows)")

    # ── 7. Print CV comparison summary for RUNLOG ──────────────────────────────
    print("\n=== CV Summary (for RUNLOG) ===")
    for name, (mean_d, std_d, _) in cv_results.items():
        tag = " ← SELECTED" if name == best_model_name else ""
        print(f"  {name}: {mean_d*1000:.0f} ms ± {std_d*1000:.0f} ms{tag}")


if __name__ == "__main__":
    main()
