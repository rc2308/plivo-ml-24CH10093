"""Ship the fixed model — retrain coherently on English + Hindi.

Fixes the red flags:
  * threads prev_pause_durs so the causal cumulative-silence features are ALIVE
    (predict.py/train.py were passing nothing -> those features were dead zeros)
  * LR + ExtraTrees soft-voting ensemble (better separated probabilities than a
    single tree that fired on everything at threshold 0.05)
  * trained on English + Hindi combined (hidden test is mostly Hindi)
  * reports the HONEST GroupKFold delay, not an in-sample number

Usage:
  python ship.py
"""
import csv, os, pickle, warnings, time
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GroupKFold

from features import load_wav, extract_features, FEATURE_NAMES, N_FEATURES

DATA = "/Users/rounak/speedrun/plivo-ml-24CH10093/eot_handout/eot_data"
TIMEOUT_S = 1.6
TH = np.round(np.arange(0.05, 1.00, 0.05), 3)
DL = np.round(np.arange(0.10, 1.65, 0.05), 3)


def score_delay(y, durs, groups, p, budget=0.05):
    y = np.asarray(y); durs = np.asarray(durs); p = np.asarray(p); groups = np.asarray(groups)
    n = len(set(groups)); ih = y == 0; ie = y == 1
    best = {"latency": TIMEOUT_S, "t": 1.0, "d": TIMEOUT_S, "cut": 0.0}
    for t in TH:
        f = p >= t; hd = durs[ih & f]; hg = groups[ih & f]; ef = f[ie]
        for d in DL:
            cut = len(set(hg[hd > d])) / max(1, n)
            if cut > budget:
                continue
            lat = np.where(ef, d, TIMEOUT_S).mean()
            if lat < best["latency"]:
                best = {"latency": float(lat), "t": float(t), "d": float(d), "cut": cut}
    return best


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p); o = np.argsort(p)
    r = np.empty(len(p), float); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def load_lang(lang):
    """Return X, y, groups, durs, keys — threading prev_pause_durs per turn (causal)."""
    dd = os.path.join(DATA, lang)
    rows = list(csv.DictReader(open(os.path.join(dd, "labels.csv"))))
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)
    cache = {}
    X, y, g, durs, keys = [], [], [], [], []
    for r in rows:
        p = os.path.join(dd, r["audio_file"])
        if p not in cache:
            cache[p] = load_wav(p)
        x, sr = cache[p]
        tid, pi = r["turn_id"], int(r["pause_index"])
        prev = [float(q["pause_end"]) - float(q["pause_start"])
                for q in by_turn[tid] if int(q["pause_index"]) < pi]
        feat = extract_features(x, sr, float(r["pause_start"]), pi, pi + 1, prev_pause_durs=prev)
        X.append(feat); y.append(1 if r["label"] == "eot" else 0)
        g.append(tid); durs.append(float(r["pause_end"]) - float(r["pause_start"]))
        keys.append((tid, pi))
    return np.array(X, np.float32), np.array(y), np.array(g), np.array(durs, np.float32), keys


def make_ensemble():
    lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                   ("clf", LogisticRegression(C=0.3, class_weight="balanced",
                                              max_iter=3000, random_state=42))])
    et = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("clf", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3,
                           class_weight="balanced", random_state=42, n_jobs=-1))])
    return VotingClassifier([("lr", lr), ("et", et)], voting="soft", weights=[1, 1])


def comb_oof(Xen, yen, gen, Xhi, yhi, ghi, n=5):
    X = np.vstack([Xen, Xhi]); y = np.concatenate([yen, yhi])
    g = np.concatenate([np.array([f"en_{x}" for x in gen]), np.array([f"hi_{x}" for x in ghi])])
    p = np.zeros(len(y))
    for tr, va in GroupKFold(n).split(X, y, g):
        m = make_ensemble(); m.fit(X[tr], y[tr]); p[va] = m.predict_proba(X[va])[:, 1]
    return p[:len(yen)], p[len(yen):]


def main():
    t0 = time.time()
    print(f"[1] Extracting {N_FEATURES} features (EN + HI, prev_pause_durs threaded)...")
    Xen, yen, gen, den, ken = load_lang("english")
    Xhi, yhi, ghi, dhi, khi = load_lang("hindi")
    print(f"    EN {Xen.shape}  HI {Xhi.shape}  in {time.time()-t0:.0f}s")

    print("[2] Honest GroupKFold delay (EN+HI trained, scored per language):")
    pen, phi = comb_oof(Xen, yen, gen, Xhi, yhi, ghi)
    ren = score_delay(yen, den, gen, pen); rhi = score_delay(yhi, dhi, ghi, phi)
    print(f"    English  OOF delay={ren['latency']*1000:.0f}ms  AUC={auc(yen,pen):.3f}  (t={ren['t']}, cut={ren['cut']*100:.1f}%)")
    print(f"    Hindi    OOF delay={rhi['latency']*1000:.0f}ms  AUC={auc(yhi,phi):.3f}  (t={rhi['t']}, cut={rhi['cut']*100:.1f}%)  <-- real target")

    print("[3] Training final LR+ET ensemble on ALL data (EN+HI)...")
    X = np.vstack([Xen, Xhi]); y = np.concatenate([yen, yhi])
    model = make_ensemble(); model.fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "model_name": "LR+ET (soft-vote)",
                     "feature_names": list(FEATURE_NAMES),
                     "honest_hindi_delay_ms": rhi["latency"]*1000,
                     "honest_hindi_auc": auc(yhi, phi)}, f)
    print("    saved -> model.pkl")

    print("[4] Writing predictions...")
    for lang, keys, X_, tag in [("english", ken, Xen, ""), ("hindi", khi, Xhi, "_hindi")]:
        p = model.predict_proba(X_)[:, 1]
        out = f"predictions{tag}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["turn_id", "pause_index", "p_eot"])
            for (tid, pi), pr in zip(keys, p):
                w.writerow([tid, pi, f"{pr:.4f}"])
        print(f"    {out}  ({len(keys)} rows)")
    # main predictions.csv = english (matches README dev loop)
    print(f"[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
