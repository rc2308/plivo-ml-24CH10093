"""Feature-subset + model search on cached features. Primary metric: Hindi OOF
delay (hidden set is mostly Hindi), with EN OOF and cross-lingual as guards."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from _eval import score_delay, auc
from _exp import load
import warnings; warnings.filterwarnings("ignore")

Xen, yen, gen, den, _, names = load("en")
Xhi, yhi, ghi, dhi, _, _ = load("hi")
names = list(names)

TRANSFER = ["pause_rate","pause_index","f0_slope_final","f0_slope_300ms",
            "f0_final_norm","f0_final_pctile","f0_final_vs_med_semitone",
            "centroid_final_norm","e_slope_last25","e_slope_last10",
            "f0_reset_semitone","e_frac_low_tail","f0_voiced_frac","rolloff_final_norm"]
PITCH = [n for n in names if n.startswith("f0_")]
PITCH_PAUSE = PITCH + ["pause_rate","pause_index","turn_dur"]

SUBSETS = {"ALL": names, "TRANSFER": TRANSFER, "PITCH": PITCH, "PITCH_PAUSE": PITCH_PAUSE}

def cols(sub): return [names.index(n) for n in sub]

def mk(kind):
    if kind == "LR":
        return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                         ("clf",LogisticRegression(C=0.3,class_weight="balanced",max_iter=3000))])
    if kind == "HGBT":
        return HistGradientBoostingClassifier(max_iter=250,max_depth=3,min_samples_leaf=20,
                 learning_rate=0.05,l2_regularization=1.0,class_weight="balanced",random_state=42)
    if kind == "RF":
        return Pipeline([("imp",SimpleImputer(strategy="median")),
                 ("clf",RandomForestClassifier(n_estimators=400,max_depth=5,min_samples_leaf=12,
                 class_weight="balanced",random_state=42,n_jobs=-1))])

def oof(X, y, g, mk_fn, n=5):
    gkf = GroupKFold(n); p = np.zeros(len(y))
    for tr,va in gkf.split(X,y,g):
        m = mk_fn(); m.fit(X[tr],y[tr]); p[va] = m.predict_proba(X[va])[:,1]
    return p

def cross(Xtr,ytr,Xte,yte,mk_fn):
    m = mk_fn(); m.fit(Xtr,ytr); return m.predict_proba(Xte)[:,1]

print(f"{'subset':<12}{'model':<6}{'HI_OOF':>8}{'HI_AUC':>8}{'EN_OOF':>8}{'EN_AUC':>8}{'EN>HI':>7}{'trAUC':>7}")
print("-"*64)
for sname, sub in SUBSETS.items():
    c = cols(sub)
    for mk_kind in ["LR","HGBT","RF"]:
        f = lambda: mk(mk_kind)
        phi = oof(Xhi[:,c], yhi, ghi, f)
        pen = oof(Xen[:,c], yen, gen, f)
        pcross = cross(Xen[:,c],yen,Xhi[:,c],yhi,f)
        hi = score_delay(yhi,dhi,ghi,phi)["latency"]*1000
        en = score_delay(yen,den,gen,pen)["latency"]*1000
        cr = score_delay(yhi,dhi,ghi,pcross)["latency"]*1000
        print(f"{sname:<12}{mk_kind:<6}{hi:>8.0f}{auc(yhi,phi):>8.3f}{en:>8.0f}{auc(yen,pen):>8.3f}{cr:>7.0f}{auc(yhi,pcross):>7.3f}")
