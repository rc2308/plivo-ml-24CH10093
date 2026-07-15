"""Extract features once and cache to .npz for fast experimentation.

Run from the starter2/ directory:
    python _cache_feats.py

Produces _feats_en.npz and _feats_hi.npz (used by _exp.py, _diag.py, _subset.py).
"""
import csv, os, sys, time
import numpy as np
from features import load_wav, extract_features, FEATURE_NAMES


def build(data_dir, tag):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))

    # Pre-build turn_pauses dict for cumulative discourse features (strictly causal)
    turn_pauses = {}
    for r in rows:
        turn_pauses.setdefault(r["turn_id"], []).append(r)

    wav_cache = {}
    X, y, groups, durs, pidx = [], [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        path = os.path.join(data_dir, r["audio_file"])
        if path not in wav_cache:
            wav_cache[path] = load_wav(path)
        x, sr = wav_cache[path]
        pi = int(r["pause_index"])
        tid = r["turn_id"]

        # Cumulative durations of ALL prior pauses in this turn (causal)
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
        durs.append(float(r["pause_end"]) - float(r["pause_start"]))
        pidx.append(pi)

    print(f"  {tag}: {len(X)} pauses, {N_FEATURES} features in {time.time()-t0:.1f}s")
    np.savez(f"_feats_{tag}.npz",
             X=np.array(X, np.float32), y=np.array(y, int),
             groups=np.array(groups), durs=np.array(durs, np.float32),
             pidx=np.array(pidx, int), names=np.array(FEATURE_NAMES))
    print(f"  Saved -> _feats_{tag}.npz  (N_FEATURES={len(FEATURE_NAMES)})")


# ── constants ─────────────────────────────────────────────────────────────────
from features import N_FEATURES

if __name__ == "__main__":
    # Run from inside starter2/ — data is one level up in eot_handout/
    DATA = os.path.join(os.path.dirname(__file__), "..", "eot_handout", "eot_data")
    build(os.path.join(DATA, "english"), "en")
    build(os.path.join(DATA, "hindi"),   "hi")
