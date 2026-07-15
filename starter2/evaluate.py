"""
Train/Test evaluation script for EOT detection.

Steps:
  1. Split turns 75% train / 25% test  (NEVER split a turn across sets)
  2. Train LR, HGBT, RF on train set
  3. Evaluate on test set → accuracy, AUC, delay metric
  4. Retrain winning model on ALL data → save model.pkl
  5. Generate final predictions.csv and predictions_hindi.csv

Usage:
    python evaluate.py \
        --data_dirs eot_handout/eot_data/english eot_handout/eot_data/hindi \
        --test_size 0.25
"""
import argparse
import csv
import os
import pickle
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import load_wav, extract_features, FEATURE_NAMES
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ─── Scorer (replicates score.py) ────────────────────────────────────────────
TIMEOUT_S  = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 3)
DELAYS     = np.round(np.arange(0.10, 1.65, 0.05), 3)


def delay_metric(meta, p_eot, budget=0.05):
    """Return best mean response delay at ≤ budget false-cutoff rate."""
    turn_ids = list({m["turn_id"] for m in meta})
    best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0, "delay": TIMEOUT_S}
    for t in THRESHOLDS:
        for d in DELAYS:
            turns_cut, latencies = set(), []
            for m, p in zip(meta, p_eot):
                if m["label"] == "hold":
                    if p >= t and d < m["pause_dur"]:
                        turns_cut.add(m["turn_id"])
                else:
                    latencies.append(d if p >= t else TIMEOUT_S)
            cutoff = len(turns_cut) / max(1, len(turn_ids))
            lat = float(np.mean(latencies)) if latencies else TIMEOUT_S
            if cutoff <= budget and lat < best["latency"]:
                best = {"latency": lat, "cutoff": cutoff, "threshold": t, "delay": d}
    return best


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(data_dirs):
    X, y, groups, meta = [], [], [], []
    cache = {}
    for data_dir in data_dirs:
        rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
        # Pre-build turn_pauses dict for cumulative discourse features (causal)
        turn_pauses = {}
        for r in rows:
            turn_pauses.setdefault(r["turn_id"], []).append(r)
        for r in rows:
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            pi = int(r["pause_index"])
            tid = r["turn_id"]
            prev_durs = [
                float(prev["pause_end"]) - float(prev["pause_start"])
                for prev in turn_pauses[tid]
                if int(prev["pause_index"]) < pi
            ]
            feat = extract_features(x, sr, float(r["pause_start"]), pi, pi + 1,
                                    prev_pause_durs=prev_durs)
            X.append(feat)
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(tid)
            meta.append({
                "turn_id":    tid,
                "pause_index": pi,
                "label":      r["label"],
                "pause_dur":  float(r["pause_end"]) - float(r["pause_start"]),
            })
    return np.array(X, dtype=np.float32), np.array(y), groups, meta


# ─── Models ───────────────────────────────────────────────────────────────────

def make_lr():
    return Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf",    LogisticRegression(C=0.1, class_weight="balanced",
                                      max_iter=2000, random_state=42)),
    ])

def make_hgbt():
    base = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, min_samples_leaf=15,
        learning_rate=0.05, l2_regularization=0.1,
        class_weight="balanced", random_state=42,
    )
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf",    CalibratedClassifierCV(base, method="isotonic", cv=5)),
    ])

def make_rf():
    base = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    return Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf",    CalibratedClassifierCV(base, method="isotonic", cv=5)),
    ])


# ─── Evaluation helper ────────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, meta_te):
    model.fit(X_tr, y_tr)
    p_te   = model.predict_proba(X_te)[:, 1]
    y_pred = (p_te >= 0.5).astype(int)

    acc    = accuracy_score(y_te, y_pred)
    auc    = roc_auc_score(y_te, p_te)
    delay  = delay_metric(meta_te, p_te)

    print(f"\n{'─'*50}")
    print(f"  Model: {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  AUC       : {auc:.3f}")
    print(f"  Delay     : {delay['latency']*1000:.0f} ms  @ threshold={delay['threshold']}, action_delay={delay['delay']*1000:.0f}ms")
    print(f"  Cutoff    : {delay['cutoff']*100:.1f}% interrupted turns")
    print(f"\n  Classification report (threshold=0.5):")
    print(classification_report(y_te, y_pred, target_names=["hold", "eot"], indent=4))

    return {"name": name, "acc": acc, "auc": auc,
            "delay_ms": delay["latency"] * 1000, "delay_result": delay,
            "model": model, "p_te": p_te}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--test_size",  type=float, default=0.25,
                    help="Fraction of TURNS held out for test (default 0.25)")
    ap.add_argument("--model_out",  default="model.pkl")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  LOADING DATA")
    print("═"*55)
    X, y, groups, meta = load_dataset(args.data_dirs)
    print(f"  Total pauses : {len(X)}")
    print(f"  EOT          : {int(y.sum())}  ({y.mean()*100:.1f}%)")
    print(f"  HOLD         : {int((y==0).sum())}  ({(1-y).mean()*100:.1f}%)")
    print(f"  Features     : {len(FEATURE_NAMES)}")
    print(f"  Unique turns : {len(set(groups))}")

    # ── 2. Turn-grouped train/test split ──────────────────────────────────────
    print("\n" + "═"*55)
    print(f"  TRAIN / TEST SPLIT  (test_size={args.test_size}, grouped by turn)")
    print("═"*55)

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr_idx, te_idx = next(gss.split(X, y, groups))

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    meta_tr = [meta[i] for i in tr_idx]
    meta_te = [meta[i] for i in te_idx]

    train_turns = len(set(m["turn_id"] for m in meta_tr))
    test_turns  = len(set(m["turn_id"] for m in meta_te))
    print(f"  Train : {len(X_tr)} pauses, {train_turns} turns  "
          f"({int(y_tr.sum())} EOT, {int((y_tr==0).sum())} HOLD)")
    print(f"  Test  : {len(X_te)} pauses, {test_turns} turns   "
          f"({int(y_te.sum())} EOT, {int((y_te==0).sum())} HOLD)")

    # ── 3. Train & evaluate all 3 models ──────────────────────────────────────
    print("\n" + "═"*55)
    print("  MODEL EVALUATION ON HELD-OUT TEST SET")
    print("═"*55)

    models = [
        ("LR",   make_lr()),
        ("HGBT", make_hgbt()),
        ("RF",   make_rf()),
    ]

    results = []
    for name, model in models:
        res = evaluate_model(name, model, X_tr, y_tr, X_te, y_te, meta_te)
        results.append(res)

    # ── 4. Summary table ──────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  SUMMARY TABLE")
    print("═"*55)
    print(f"  {'Model':<8}  {'Accuracy':>10}  {'AUC':>6}  {'Delay (ms)':>12}  {'Cutoff %':>10}")
    print(f"  {'-'*52}")
    for r in results:
        print(f"  {r['name']:<8}  {r['acc']*100:>9.1f}%  {r['auc']:>6.3f}  "
              f"{r['delay_ms']:>12.0f}  {r['delay_result']['cutoff']*100:>9.1f}%")

    best = min(results, key=lambda r: r["delay_ms"])
    print(f"\n  ► Best model (lowest delay): {best['name']} — {best['delay_ms']:.0f} ms")

    # ── 5. Retrain best model on ALL data ─────────────────────────────────────
    print("\n" + "═"*55)
    print(f"  RETRAINING {best['name']} ON ALL DATA ({len(X)} pauses)")
    print("═"*55)

    # Rebuild fresh model (not the one fitted on train split)
    final_model_map = {"LR": make_lr, "HGBT": make_hgbt, "RF": make_rf}
    final_model = final_model_map[best["name"]]()
    final_model.fit(X, y)

    artifact = {
        "model":         final_model,
        "model_name":    best["name"],
        "test_accuracy": best["acc"],
        "test_auc":      best["auc"],
        "test_delay_ms": best["delay_ms"],
        "feature_names": FEATURE_NAMES,
    }
    with open(args.model_out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  Saved -> {args.model_out}")
    print(f"  Test accuracy : {best['acc']*100:.1f}%")
    print(f"  Test AUC      : {best['auc']:.3f}")
    print(f"  Test delay    : {best['delay_ms']:.0f} ms")
    print(f"\n  Now run:")
    print(f"  python predict.py --data_dir eot_handout/eot_data/english --out predictions.csv")
    print(f"  python score.py   --data_dir eot_handout/eot_data/english --pred predictions.csv")
    print(f"  python predict.py --data_dir eot_handout/eot_data/hindi   --out predictions_hindi.csv")
    print(f"  python score.py   --data_dir eot_handout/eot_data/hindi   --pred predictions_hindi.csv")


if __name__ == "__main__":
    main()
