"""Per-feature univariate diagnostic: which features carry transferable signal?"""
import numpy as np
from _eval import auc
from _exp import load

Xen, yen, *_ , names = load("en")
Xhi, yhi, *_ = load("hi")[:3] + (load("hi")[-1],)
Xhi, yhi = load("hi")[0], load("hi")[1]

def uni_auc(X, y):
    out = []
    for j in range(X.shape[1]):
        col = X[:, j].astype(float)
        m = ~np.isnan(col)
        if m.sum() < 10:
            out.append(np.nan); continue
        a = auc(y[m], col[m])
        out.append(max(a, 1 - a))  # direction-agnostic strength
    return np.array(out)

aen = uni_auc(Xen, yen)
ahi = uni_auc(Xhi, yhi)
# signed AUC (does higher value -> EOT?) to check sign consistency
def signed(X, y):
    out = []
    for j in range(X.shape[1]):
        col = X[:, j].astype(float); m = ~np.isnan(col)
        out.append(auc(y[m], col[m]) if m.sum() >= 10 else np.nan)
    return np.array(out)
sen, shi = signed(Xen, yen), signed(Xhi, yhi)

print(f"{'feature':<26}{'EN|auc|':>9}{'HI|auc|':>9}{'EN_sgn':>8}{'HI_sgn':>8}{'consist':>9}")
print("-"*79)
order = np.argsort(-(np.nan_to_num(aen)+np.nan_to_num(ahi)))
for j in order:
    consist = "OK" if (sen[j]-0.5)*(shi[j]-0.5) > 0 else "FLIP"
    print(f"{names[j]:<26}{aen[j]:>9.3f}{ahi[j]:>9.3f}{sen[j]:>8.3f}{shi[j]:>8.3f}{consist:>9}")
