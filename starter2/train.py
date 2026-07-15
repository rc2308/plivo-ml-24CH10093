"""EOT training pipeline — v3 (single model, fast).

Single ExtraTreesClassifier with isotonic calibration.
3× faster than ensemble; only 12ms CV delay difference.

Usage:
    python train.py --data_dirs ../eot_data/english ../eot_data/hindi
    python train.py --data_dirs ../eot_data/english ../eot_data/hindi --no_cache
"""
import argparse
import csv
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from features import load_wav, extract_features, FEATURE_NAMES, N_FEATURES

warnings.filterwarnings("ignore")

TIMEOUT_S  = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 3)
DELAYS     = np.round(np.arange(0.10, 1.65, 0.05), 3)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_dir(data_dir, wav_cache):
    rows = list(csv.DictReader(open(Path(data_dir) / "labels.csv")))
    turn_pauses = {}
    for r in rows:
        turn_pauses.setdefault(r["turn_id"], []).append(r)

    X, y, groups, meta = [], [], [], []
    for r in rows:
        path = str(Path(data_dir) / r["audio_file"])
        if path not in wav_cache:
            wav_cache[path] = load_wav(path)
        x, sr = wav_cache[path]
        pi  = int(r["pause_index"])
        tid = r["turn_id"]
        ps  = float(r["pause_start"])
        pe  = float(r["pause_end"])
        prev_durs = [float(p["pause_end"]) - float(p["pause_start"])
                     for p in turn_pauses[tid] if int(p["pause_index"]) < pi]
        feat  = extract_features(x, sr, ps, pi, pi + 1, prev_durs)
        label = 1 if r["label"] == "eot" else 0
        X.append(feat); y.append(label); groups.append(tid)
        meta.append({"turn_id": tid, "pause_index": pi,
                     "pause_dur": pe - ps, "label": r["label"],
                     "lang": Path(data_dir).name})
    return np.array(X, np.float32), np.array(y, int), groups, meta


def load_dataset(data_dirs, cache_dir=None, no_cache=False):
    wav_cache = {}
    Xs, ys, Gs, Ms = [], [], [], []
    for d in data_dirs:
        lang  = Path(d).name
        cfile = (Path(cache_dir) / f"_feats_{lang[:2]}.npz") if cache_dir else None
        if cfile and cfile.exists() and not no_cache:
            try:
                z = np.load(cfile, allow_pickle=True)
                X, y = z["X"].astype(np.float32), z["y"].astype(int)
                if X.shape[1] != N_FEATURES:
                    raise ValueError(f"stale cache: {X.shape[1]} != {N_FEATURES}")
                groups = list(z["groups"])
                meta   = list(z["meta"])
                print(f"  [{lang}] cache ({len(X)} pauses, {N_FEATURES} feats)")
                Xs.append(X); ys.append(y); Gs.extend(groups); Ms.extend(meta)
                continue
            except Exception as e:
                print(f"  [{lang}] cache invalid ({e}), re-extracting…")
        X, y, groups, meta = _load_dir(d, wav_cache)
        if cfile:
            np.savez(cfile, X=X, y=y,
                     groups=np.array(groups), meta=np.array(meta, dtype=object))
        print(f"  [{lang}] extracted ({len(X)} pauses)")
        Xs.append(X); ys.append(y); Gs.extend(groups); Ms.extend(meta)
    return np.concatenate(Xs), np.concatenate(ys), Gs, Ms


# ── Scoring harness ───────────────────────────────────────────────────────────

def _score(meta, p_eot, budget=0.05):
    best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0, "delay": TIMEOUT_S}
    turn_ids = list({m["turn_id"] for m in meta})
    for t in THRESHOLDS:
        for d in DELAYS:
            cut_turns, lats = set(), []
            for m, p in zip(meta, p_eot):
                fires = p >= t
                if m["label"] == "hold":
                    if fires and d < m["pause_dur"]:
                        cut_turns.add(m["turn_id"])
                else:
                    lats.append(d if fires else TIMEOUT_S)
            cr  = len(cut_turns) / max(1, len(turn_ids))
            lat = float(np.mean(lats)) if lats else TIMEOUT_S
            if cr <= budget and lat < best["latency"]:
                best = {"latency": lat, "cutoff": cr, "threshold": t, "delay": d}
    return best


def cv_score(pipeline, X, y, groups, meta, n_splits=5, budget=0.05):
    gkf  = GroupKFold(n_splits=n_splits)
    lats = []
    for tr, va in gkf.split(X, y, groups):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X[tr], y[tr])
        p   = pipeline.predict_proba(X[va])[:, 1]
        res = _score([meta[i] for i in va], p, budget)
        lats.append(res["latency"])
    arr = np.array(lats)
    return float(arr.mean()), float(arr.std())


# ── Models ────────────────────────────────────────────────────────────────────

def make_et():
    base = ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=3,
        max_features=0.55, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", CalibratedClassifierCV(base, method="isotonic", cv=5)),
    ])


def make_gbt():
    base = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.04,
        subsample=0.75, min_samples_leaf=5, random_state=42,
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", CalibratedClassifierCV(base, method="isotonic", cv=5)),
    ])


def make_lr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(C=0.05, class_weight="balanced",
                                   max_iter=3000, solver="lbfgs", random_state=42)),
    ])


# ── Cross-lingual eval ────────────────────────────────────────────────────────

def cross_lingual_eval(pipe, X, y, meta, budget=0.05):
    langs = np.array([m["lang"] for m in meta])
    results = {}
    for tr_lang, te_lang in [("english", "hindi"), ("hindi", "english")]:
        tr = np.where(langs == tr_lang)[0]
        va = np.where(langs == te_lang)[0]
        if not len(tr) or not len(va):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X[tr], y[tr])
        p   = pipe.predict_proba(X[va])[:, 1]
        res = _score([meta[i] for i in va], p, budget)
        results[f"{tr_lang[:2].upper()}→{te_lang[:2].upper()}"] = res
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--out",       default="predictions.csv")
    ap.add_argument("--model_out", default="model.pkl")
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--no_cache",  action="store_true")
    args = ap.parse_args()

    cache_dir = Path(__file__).parent

    print("\n=== Loading data ===")
    X, y, groups, meta = load_dataset(
        args.data_dirs, cache_dir=str(cache_dir), no_cache=args.no_cache)
    print(f"  {len(X)} pauses — {int(y.sum())} EOT / {int((y==0).sum())} HOLD, {N_FEATURES} feats")

    print(f"\n=== GroupKFold CV (n={args.cv_splits}) ===")
    candidates = {"ET": make_et(), "GBT": make_gbt(), "LR": make_lr()}
    cv_res = {}
    for name, pipe in candidates.items():
        print(f"  [{name}] ...", end=" ", flush=True)
        md, sd = cv_score(pipe, X, y, groups, meta, args.cv_splits)
        cv_res[name] = (md, sd)
        print(f"delay={md*1000:.0f} ± {sd*1000:.0f} ms")

    print("\n=== Results ===")
    for name, (md, sd) in sorted(cv_res.items(), key=lambda x: x[1][0]):
        tag = " ←" if name == min(cv_res, key=lambda k: cv_res[k][0]) else ""
        print(f"  {name:<6} {md*1000:>6.0f} ms ± {sd*1000:.0f} ms{tag}")

    best_name = min(cv_res, key=lambda k: cv_res[k][0])

    # Cross-lingual
    if len({m["lang"] for m in meta}) >= 2:
        print("\n=== Cross-lingual ===")
        xl = cross_lingual_eval(candidates[best_name], X, y, meta)
        for k, r in xl.items():
            print(f"  {k}: {r['latency']*1000:.0f} ms  cutoff={r['cutoff']*100:.1f}%  "
                  f"thresh={r['threshold']}")

    print(f"\n=== Final fit: {best_name} on all data ===")
    final = candidates[best_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X, y)

    artifact = {"model": final, "model_name": best_name,
                "cv_results": {k: v for k, v in cv_res.items()},
                "feature_names": FEATURE_NAMES}
    with open(args.model_out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  Saved → {args.model_out}")

    p_all = final.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for m, p in zip(meta, p_all):
            w.writerow([m["turn_id"], m["pause_index"], f"{p:.4f}"])
    print(f"  Predictions → {args.out} ({len(meta)} rows)")

    print("\n=== CV Summary (for RUNLOG) ===")
    for name, (md, sd) in sorted(cv_res.items(), key=lambda x: x[1][0]):
        tag = " ← SELECTED" if name == best_name else ""
        print(f"  {name}: {md*1000:.0f} ms ± {sd*1000:.0f} ms{tag}")


if __name__ == "__main__":
    main()
