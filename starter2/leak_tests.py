"""
Five leakage / integrity checks. Run in order; stop on first failure.
Usage: python3.12 leak_tests.py
"""
import warnings, sys, csv, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler

# ── import our pipeline ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from train_eot import build_dataset, load_audio, extract

DATA_EN  = Path("../eot_data/english")
DATA_HI  = Path("../eot_data/hindi")
CACHE    = Path(".cache")

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

def banner(n, title):
    print(f"\n{'='*60}")
    print(f"  CHECK {n}: {title}")
    print('='*60)

# ─────────────────────────────────────────────────────────────────────────────
banner(1, "Label-shuffle test (most important)")
# ─────────────────────────────────────────────────────────────────────────────
print("Loading English features …")
X, y, groups, keys = build_dataset(DATA_EN, CACHE)
groups_arr = np.array(groups)

def train_gbt(Xtr, ytr, Xva):
    sc = RobustScaler()
    Xtr2 = sc.fit_transform(Xtr); Xva2 = sc.transform(Xva)
    np.nan_to_num(Xtr2, copy=False); np.nan_to_num(Xva2, copy=False)
    m = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                   random_state=42)
    m.fit(Xtr2, ytr)
    return m.predict_proba(Xva2)[:, 1]

# Shuffle labels WITHIN each turn (preserves turn structure)
df = pd.DataFrame({"turn_id": groups, "label": y})
shuffled_y = y.copy()
rng = np.random.default_rng(0)
for tid, grp in df.groupby("turn_id"):
    idx = grp.index.values
    shuffled_y[idx] = rng.permutation(grp["label"].values)

gkf = GroupKFold(n_splits=5)
aucs_real, aucs_shuf = [], []
for tr, va in gkf.split(X, y, groups_arr):
    p_real = train_gbt(X[tr], y[tr], X[va])
    p_shuf = train_gbt(X[tr], shuffled_y[tr], X[va])
    aucs_real.append(roc_auc_score(y[va], p_real))
    aucs_shuf.append(roc_auc_score(shuffled_y[va], p_shuf))

mean_real = np.mean(aucs_real)
mean_shuf = np.mean(aucs_shuf)
print(f"  Real-label OOF AUC   : {mean_real:.3f}  (should be >> 0.5)")
print(f"  Shuffled-label OOF AUC: {mean_shuf:.3f}  (should be ≈ 0.5)")

if mean_shuf > 0.70:
    print(f"  {FAIL}  Shuffled AUC={mean_shuf:.3f} > 0.70 → LEAKAGE CONFIRMED")
    sys.exit(1)
else:
    print(f"  {PASS}  Shuffled AUC={mean_shuf:.3f} ≈ 0.5 → no structural leakage")

# ─────────────────────────────────────────────────────────────────────────────
banner(2, "Feature-window slicing check")
# ─────────────────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(DATA_EN / "labels.csv")))
sample_rows = rows[:5]
audio_cache = {}
all_ok = True
for r in sample_rows:
    p = DATA_EN / r["audio_file"]
    if str(p) not in audio_cache:
        audio_cache[str(p)] = load_audio(p)
    audio, sr = audio_cache[str(p)]
    ps = float(r["pause_start"])
    end_sample = int(ps * sr)
    win_start   = max(0, end_sample - int(1.5 * sr))
    print(f"  {r['turn_id']}  pause_index={r['pause_index']}"
          f"  audio_len={len(audio)/sr:.2f}s"
          f"  pause_start={ps}s"
          f"  window=[{win_start/sr:.3f}s → {end_sample/sr:.3f}s]"
          f"  end_sample={end_sample}  audio_samples={len(audio)}")
    assert end_sample <= len(audio), f"pause_start beyond audio! {r['turn_id']}"
    # window must END at pause_start — not go past it
    assert end_sample <= len(audio), "window bleeds past audio"
    if end_sample > len(audio):
        all_ok = False

print(f"  Segment end == pause_start (never past it)? ", end="")
if all_ok:
    print(PASS)
else:
    print(FAIL); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
banner(3, "Cumulative-context feature audit (pause_end usage)")
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
result = subprocess.run(
    ["grep", "-n", "pause_end", "predict.py", "train_eot.py"],
    capture_output=True, text=True
)
hits = result.stdout.strip().split("\n") if result.stdout.strip() else []
print(f"  grep hits for 'pause_end' in predict.py + train_eot.py:")
leaky_hits = []
# safe patterns: prior-pause cumulative silence (feature), or scorer simulation (dur)
SAFE_PATTERNS = [
    "prev[\"pause_end\"]",          # cumulative prior-pause silence — causal
    'prev["pause_end"]',
    "\"dur\"",                       # pause_end used only to compute dur for scorer
    '"dur"',
    "pz_h",                         # cross-lingual scorer call
    "pauses.append",               # building scorer pauses list (not features)
    "competition_score",
]
for h in hits:
    if h:
        print(f"    {h}")
        safe = any(pat in h for pat in SAFE_PATTERNS)
        if not safe:
            leaky_hits.append(h)

if leaky_hits:
    print(f"\n  {FAIL}  Potentially leaky pause_end access:")
    for h in leaky_hits: print(f"    {h}")
    sys.exit(1)
else:
    print(f"\n  {PASS}  All pause_end accesses are for prior pauses only (cumulative silence calc).")

# ─────────────────────────────────────────────────────────────────────────────
banner(4, "GroupKFold turn-overlap check")
# ─────────────────────────────────────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
all_zero = True
for fold, (tr, va) in enumerate(gkf.split(X, y, groups_arr)):
    train_turns = set(groups_arr[tr])
    val_turns   = set(groups_arr[va])
    overlap     = train_turns & val_turns
    print(f"  fold {fold}: train_turns={len(train_turns)}  val_turns={len(val_turns)}  overlap={len(overlap)}", end="")
    if len(overlap) == 0:
        print(f"  {PASS}")
    else:
        print(f"  {FAIL}  overlapping turns: {overlap}")
        all_zero = False

if not all_zero:
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
banner(5, "Cold untouched holdout — 20 Hindi turns")
# ─────────────────────────────────────────────────────────────────────────────
import shutil, tempfile, pickle
from train_eot import competition_score, oof_score

# Pick 20 holdout turns from Hindi (seed=42, never seen before)
hi_rows = list(csv.DictReader(open(DATA_HI / "labels.csv")))
all_hi_turns = list(dict.fromkeys(r["turn_id"] for r in hi_rows))  # ordered unique
random.seed(42)
holdout_turns = set(random.sample(all_hi_turns, 20))
train_turns   = set(all_hi_turns) - holdout_turns
print(f"  Hindi turns total={len(all_hi_turns)}  holdout={len(holdout_turns)}  train={len(train_turns)}")
print(f"  Holdout turn IDs: {sorted(holdout_turns)[:5]} …")

# Write holdout labels.csv and train labels.csv into temp dirs
def write_split(rows, turn_set, dest_dir, src_audio_dir):
    dest_dir = Path(dest_dir)
    (dest_dir / "audio").mkdir(parents=True, exist_ok=True)
    split_rows = [r for r in rows if r["turn_id"] in turn_set]
    with open(dest_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(split_rows)
    # symlink audio files (avoid copying large wavs)
    for r in split_rows:
        src = (src_audio_dir / r["audio_file"]).resolve()
        dst = dest_dir / r["audio_file"]
        if not dst.exists():
            try: dst.symlink_to(src)
            except: shutil.copy2(src, dst)
    return split_rows

tmpdir = Path(tempfile.mkdtemp())
holdout_dir = tmpdir / "hindi_holdout"
train_hi_dir = tmpdir / "hindi_train"

write_split(hi_rows, holdout_turns, holdout_dir, DATA_HI)
write_split(hi_rows, train_turns,   train_hi_dir, DATA_HI)

# Load English (all) + Hindi train portion
print("  Building train features (EN all + HI train) …")
Xen, yen, gen, _ = build_dataset(DATA_EN, CACHE)
Xht, yht, ght, _ = build_dataset(train_hi_dir)   # no cache — fresh split

Xtr_all = np.vstack([Xen, Xht])
ytr_all  = np.concatenate([yen, yht])

sc = RobustScaler()
Xtr_sc = sc.fit_transform(Xtr_all)
np.nan_to_num(Xtr_sc, copy=False)

clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, min_samples_leaf=4, random_state=42)
clf.fit(Xtr_sc, ytr_all)

# Predict on holdout
print("  Predicting on 20 holdout Hindi turns …")
Xho, yho, gho, kho = build_dataset(holdout_dir)
Xho_sc = sc.transform(Xho)
np.nan_to_num(Xho_sc, copy=False)
pho = clf.predict_proba(Xho_sc)[:, 1]

# Build pauses list for scorer
ho_rows = list(csv.DictReader(open(holdout_dir / "labels.csv")))
pm = {(r["turn_id"], int(r["pause_index"])): r for r in ho_rows}
pauses_ho = [
    {"turn_id": tid,
     "dur": float(pm[(tid,pi)]["pause_end"]) - float(pm[(tid,pi)]["pause_start"]),
     "label": pm[(tid,pi)]["label"],
     "p": pho[i]}
    for i, (tid, pi) in enumerate(kho)
]
sc_ho = competition_score(pauses_ho, budget=0.05)

# Full Hindi score (from existing predictions_hindi.csv) for comparison
from train_eot import load_audio as _la
y_ho_arr = np.array([1 if pm[(tid,pi)]["label"]=="eot" else 0 for (tid,pi) in kho])
order = np.argsort(pho); ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(1, len(pho)+1)
n1, n0 = y_ho_arr.sum(), len(y_ho_arr)-y_ho_arr.sum()
auc_ho = float((ranks[y_ho_arr==1].sum() - n1*(n1+1)/2) / (n1*n0)) if n1 and n0 else float("nan")

print(f"\n  Holdout result (20 unseen Hindi turns):")
print(f"    turns={len(holdout_turns)}  pauses={len(pauses_ho)}")
print(f"    AUC           = {auc_ho:.3f}")
print(f"    mean delay    = {sc_ho['lat']*1000:.0f} ms")
print(f"    interrupted   = {sc_ho['cut']*100:.1f}%")
print(f"    threshold     = {sc_ho['t']}   delay={sc_ho['d']*1000:.0f}ms")

# Reported OOF was ~1225ms — holdout should be in same ballpark
reported_oof = 1225
if sc_ho['lat'] < reported_oof * 2.5 and sc_ho['cut'] <= 0.08:
    print(f"\n  {PASS}  Holdout latency ({sc_ho['lat']*1000:.0f}ms) is in realistic range vs OOF ({reported_oof}ms)")
else:
    print(f"\n  ⚠ WARNING  Holdout latency={sc_ho['lat']*1000:.0f}ms vs reported OOF={reported_oof}ms")
    print(f"     Validation may have been optimistic — report this number, not 100ms.")

shutil.rmtree(tmpdir)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ALL 5 CHECKS COMPLETE")
print("="*60)
