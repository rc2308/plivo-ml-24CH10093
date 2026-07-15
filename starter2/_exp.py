"""Run honest evaluation experiments on cached features."""
import sys
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from _eval import score_delay, auc

import warnings; warnings.filterwarnings("ignore")


def load(tag):
    d = np.load(f"_feats_{tag}.npz", allow_pickle=True)
    return d["X"], d["y"], d["groups"], d["durs"], d["pidx"], list(d["names"])


def make_lr():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", RobustScaler()),
                     ("clf", LogisticRegression(C=0.3, class_weight="balanced",
                                                max_iter=3000))])

def make_hgbt():
    return Pipeline([
        ("sc", RobustScaler()),
        ("clf", CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_iter=300, max_depth=3, min_samples_leaf=20,
                learning_rate=0.04, l2_regularization=1.0,
                class_weight="balanced", random_state=42),
            method="isotonic", cv=5))])

def make_rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", RobustScaler()),
                     ("clf", CalibratedClassifierCV(
                         RandomForestClassifier(
                             n_estimators=400, max_depth=5, min_samples_leaf=12,
                             class_weight="balanced", random_state=42, n_jobs=-1),
                         method="isotonic", cv=5))])

def make_et():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", RobustScaler()),
                     ("clf", CalibratedClassifierCV(
                         ExtraTreesClassifier(
                             n_estimators=300, min_samples_leaf=3,
                             class_weight="balanced", random_state=42, n_jobs=-1),
                         method="isotonic", cv=5))])

def make_gbt():
    """GBT — winner in starter codebase (autocorr features); test with pyin features."""
    return Pipeline([
        ("sc", RobustScaler()),
        ("clf", CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=4, random_state=42),
            method="isotonic", cv=5))])

MODELS = {"LR": make_lr, "HGBT": make_hgbt, "RF": make_rf, "ET": make_et, "GBT": make_gbt}


def cv_within(tag, model_fn, n_splits=5):
    X, y, g, durs, pidx, names = load(tag)
    gkf = GroupKFold(n_splits=n_splits)
    lats, aucs = [], []
    oof = np.zeros(len(y))
    for tr, va in gkf.split(X, y, g):
        m = model_fn(); m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        oof[va] = p
        r = score_delay(y[va], durs[va], g[va], p)
        lats.append(r["latency"]); aucs.append(auc(y[va], p))
    # pooled OOF score
    pooled = score_delay(y, durs, g, oof)
    return np.mean(lats)*1000, np.std(lats)*1000, np.mean(aucs), pooled["latency"]*1000, auc(y, oof)


def cross(train_tag, test_tag, model_fn):
    Xtr, ytr, gtr, _, _, _ = load(train_tag)
    Xte, yte, gte, dte, _, _ = load(test_tag)
    m = model_fn(); m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    r = score_delay(yte, dte, gte, p)
    return r["latency"]*1000, auc(yte, p), r


def combined_cv(model_fn, n_splits=5):
    """Train on EN+HI combined, grouped CV — mirrors final model."""
    Xen, yen, gen, den, _, _ = load("en")
    Xhi, yhi, ghi, dhi, _, _ = load("hi")
    X = np.vstack([Xen, Xhi]); y = np.concatenate([yen, yhi])
    g = np.concatenate([np.array([f"en_{x}" for x in gen]),
                        np.array([f"hi_{x}" for x in ghi])])
    durs = np.concatenate([den, dhi])
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    for tr, va in gkf.split(X, y, g):
        m = model_fn(); m.fit(X[tr], y[tr]); oof[va] = m.predict_proba(X[va])[:, 1]
    # score EN and HI separately from the OOF predictions
    n_en = len(yen)
    r_en = score_delay(yen, den, gen, oof[:n_en])
    r_hi = score_delay(yhi, dhi, ghi, oof[n_en:])
    return r_en, r_hi, auc(yen, oof[:n_en]), auc(yhi, oof[n_en:])


if __name__ == "__main__":
    print("="*70)
    print("WITHIN-LANGUAGE GroupKFold CV (mean±std over folds | pooled-OOF)")
    print("="*70)
    for tag in ["en", "hi"]:
        print(f"\n--- {tag.upper()} ---")
        for name, fn in MODELS.items():
            m, s, a, pl, pa = cv_within(tag, fn)
            print(f"  {name:5s}  fold delay={m:4.0f}±{s:3.0f}ms  foldAUC={a:.3f}  | OOF delay={pl:4.0f}ms  OOF_AUC={pa:.3f}")

    print("\n" + "="*70)
    print("CROSS-LINGUAL TRANSFER (the real grade proxy)")
    print("="*70)
    for tr, te in [("en", "hi"), ("hi", "en")]:
        print(f"\n--- train {tr.upper()} -> test {te.upper()} ---")
        for name, fn in MODELS.items():
            lat, a, r = cross(tr, te, fn)
            print(f"  {name:5s}  delay={lat:4.0f}ms  AUC={a:.3f}  (thr={r['threshold']}, d={r['delay']*1000:.0f}ms, cut={r['cutoff']*100:.1f}%)")

    print("\n" + "="*70)
    print("COMBINED EN+HI, grouped OOF (mirrors deployed model)")
    print("="*70)
    for name, fn in MODELS.items():
        r_en, r_hi, aen, ahi = combined_cv(fn)
        print(f"  {name:5s}  EN OOF delay={r_en['latency']*1000:4.0f}ms (AUC {aen:.3f})  |  HI OOF delay={r_hi['latency']*1000:4.0f}ms (AUC {ahi:.3f})")
