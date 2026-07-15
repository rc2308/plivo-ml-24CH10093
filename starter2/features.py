"""EOT feature extraction — strict causal, cross-lingual, single-pyin.

CAUSALITY CONTRACT
    extract_features() slices x[0 : int(pause_start*sr)] ONCE, at the top,
    before any downstream call. No helper ever sees audio at/after pause_start.
    (Verified by reading this file: every family receives `ac` = the causal slice.)

DESIGN
    The transferable end-of-turn cue is turn-FINAL prosody, and it must be
    expressed RELATIVE to the speaker's own turn — raw Hz/dB do not transfer
    across speakers or languages. So almost every feature here is a slope, a
    ratio, or a value normalized against this turn's own distribution.

    Pitch (pyin) is computed exactly ONCE, on the last PITCH_WIN_S seconds of
    causal audio (turn-final prosody lives there), then reused everywhere.
    This is both faster and where the signal is.
"""
import warnings

import librosa
import numpy as np
import soundfile as sf

SR_HOP_MS = 10          # frame hop for RMS / pitch
FRAME_MS = 25
PITCH_WIN_S = 2.5       # pyin analysis window (turn-final prosody)
FMIN, FMAX = 60.0, 400.0


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: str):
    """Load WAV to float32 mono array + sample rate."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def causal_slice(x: np.ndarray, sr: int, pause_start: float) -> np.ndarray:
    """x[0 : pause_start*sr] — the ONLY audio any feature code sees."""
    end = int(round(pause_start * sr))
    end = min(end, len(x))
    return x[:end]


# ─────────────────────────────────────────────────────────────────────────────
# Shared low-level computations (done once, reused across families)
# ─────────────────────────────────────────────────────────────────────────────

def _tail(a: np.ndarray, sr: int, sec: float) -> np.ndarray:
    n = int(sec * sr)
    return a[max(0, len(a) - n):]


def _rms(a: np.ndarray, sr: int) -> np.ndarray:
    fl = int(sr * FRAME_MS / 1000)
    hp = int(sr * SR_HOP_MS / 1000)
    if len(a) < fl:
        return np.array([np.sqrt(np.mean(a ** 2) + 1e-12)], np.float32)
    n = 1 + (len(a) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return np.sqrt(np.mean(a[idx] ** 2, axis=1) + 1e-12).astype(np.float32)


def _pyin(a: np.ndarray, sr: int):
    """Single pyin pass. Returns (f0 with NaN unvoiced, voiced_flag)."""
    hop = int(sr * SR_HOP_MS / 1000)
    if len(a) < hop * 4:
        return np.array([np.nan], np.float32), np.array([False])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, vflag, _ = librosa.pyin(a.astype(np.float32), fmin=FMIN, fmax=FMAX,
                                    sr=sr, hop_length=hop, fill_na=np.nan)
    return f0.astype(np.float32), vflag.astype(bool)


def _voiced_runs(vf: np.ndarray):
    segs, inv, start = [], False, 0
    for i, v in enumerate(vf):
        if v and not inv:
            start, inv = i, True
        elif not v and inv:
            segs.append((start, i)); inv = False
    if inv:
        segs.append((start, len(vf)))
    return segs


def _slope(v: np.ndarray) -> float:
    """OLS slope over index; NaN if <3 valid points. Scaled to per-second-ish."""
    valid = v[~np.isnan(v)]
    if len(valid) < 3:
        return np.nan
    idx = np.where(~np.isnan(v))[0].astype(float)
    s, _ = np.polyfit(idx, valid, 1)
    return float(s)


def _pct_rank(value, arr) -> float:
    """Percentile rank (0..1) of `value` within valid entries of arr."""
    a = arr[~np.isnan(arr)]
    if len(a) < 3 or np.isnan(value):
        return np.nan
    return float((a < value).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Feature families — all receive the causal slice `ac` and precomputed arrays
# ─────────────────────────────────────────────────────────────────────────────

def _pitch_feats(ac, sr, f0, vf) -> dict:
    """Turn-final F0 dynamics, normalized to the speaker's own range."""
    out = {}
    voiced = f0[~np.isnan(f0)]

    out["f0_voiced_frac"] = float(vf.mean()) if len(vf) else np.nan
    near = vf[-15:] if len(vf) >= 15 else vf
    out["f0_voiced_frac_near"] = float(near.mean()) if len(near) else np.nan

    if len(voiced) >= 4:
        med = float(np.median(voiced))
        p10, p90 = np.percentile(voiced, [10, 90])
        rng = max(p90 - p10, 1e-6)
        last = voiced[-min(5, len(voiced)):]
        last_m = float(last.mean())
        # where does the final pitch sit in the speaker's own range? (low=terminal)
        out["f0_final_norm"] = (last_m - med) / rng
        out["f0_final_pctile"] = _pct_rank(last_m, voiced)
        out["f0_final_vs_med_semitone"] = 12.0 * np.log2((last_m + 1e-6) / (med + 1e-6))
        # trajectory slopes over the last voiced frames (falling = EOT)
        n = min(12, len(voiced))
        out["f0_slope_final"] = _slope(voiced[-n:]) / (med + 1e-6)  # relative slope
        out["f0_range_norm"] = rng / (med + 1e-6)
    else:
        for k in ["f0_final_norm", "f0_final_pctile", "f0_final_vs_med_semitone",
                  "f0_slope_final", "f0_range_norm"]:
            out[k] = np.nan

    # slope over the final 300 ms (raw frames)
    f0_t, _ = _pyin(_tail(ac, sr, 0.3), sr)
    med_all = float(np.nanmedian(f0)) if np.isfinite(np.nanmedian(f0)) else 1.0
    out["f0_slope_300ms"] = _slope(f0_t) / (med_all + 1e-6)

    # continuation rise vs final fall across last two voiced runs (pitch reset)
    runs = _voiced_runs(vf)
    if len(runs) >= 2:
        (s1, e1), (s2, e2) = runs[-1], runs[-2]
        a = f0[s1:e1]; b = f0[s2:e2]
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if len(a) and len(b):
            out["f0_reset_semitone"] = 12.0 * np.log2((a.mean() + 1e-6) / (b.mean() + 1e-6))
        else:
            out["f0_reset_semitone"] = np.nan
    else:
        out["f0_reset_semitone"] = np.nan
    return out


def _energy_feats(ac, sr, rms) -> dict:
    """RMS decay/taper, normalized to the turn's own energy range."""
    out = {}
    db = 20 * np.log10(rms + 1e-12)
    lo, hi = np.percentile(db, [5, 95])
    rng = max(hi - lo, 1e-6)
    out["e_std_norm"] = float(db.std() / rng)
    # final energy position within the turn's range (low = trailing to silence)
    tailn = db[-8:] if len(db) >= 8 else db
    out["e_final_norm"] = float((tailn.mean() - np.median(db)) / rng)
    out["e_final_pctile"] = _pct_rank(float(tailn.mean()), db)
    # decay slopes (dB/frame) normalized by range
    out["e_slope_last10"] = (_slope(db[-10:]) / rng) if len(db) >= 10 else np.nan
    out["e_slope_last25"] = (_slope(db[-25:]) / rng) if len(db) >= 25 else np.nan
    # ratio of final RMS to turn mean (linear; <1 = decaying)
    out["e_ratio_tail"] = float(rms[-5:].mean() / (rms.mean() + 1e-12)) if len(rms) >= 5 else np.nan
    # how much of the last 300ms is below a low-energy (near-silence) gate
    gate = lo + 0.15 * rng
    last30 = db[-30:] if len(db) >= 30 else db
    out["e_frac_low_tail"] = float((last30 < gate).mean())
    # full-turn energy arc features (ported from starter's 42-feature set)
    n50 = max(1, int(500 / SR_HOP_MS))  # ~50 frames = 500 ms
    tail50 = db[-n50:] if len(db) >= n50 else db
    out["e_slope_last50"] = (_slope(tail50) / rng) if len(tail50) >= 3 else np.nan
    out["e_total_drop"] = float((db[-1] - db[0]) / rng) if len(db) >= 2 else np.nan
    if len(db) >= n50:
        out["e_full_pct_drop"] = float(
            np.percentile(db[-n50:], 25) - np.percentile(db, 25)) / rng
    else:
        out["e_full_pct_drop"] = np.nan
    return out


def _timing_feats(ac, sr, pause_start, pause_index, vf) -> dict:
    """Turn structure + final lengthening + terminal slowing."""
    out = {}
    out["turn_dur"] = float(pause_start)
    out["log_turn_dur"] = float(np.log1p(pause_start))
    out["pause_index"] = float(pause_index)
    out["pause_rate"] = float(pause_index) / (float(pause_start) / 60.0 + 1e-6)

    runs = _voiced_runs(vf)
    hop_s = SR_HOP_MS / 1000.0
    if runs:
        durs = np.array([(e - s) * hop_s for s, e in runs])
        out["last_voiced_dur"] = float(durs[-1])
        if len(durs) >= 2:
            # final lengthening: last segment vs median of prior segments
            out["final_leng_ratio"] = float(durs[-1] / (np.median(durs[:-1]) + 1e-6))
        else:
            out["final_leng_ratio"] = np.nan
        # trailing gap: voiced frames stopped this long before pause onset
        out["trail_unvoiced_s"] = float((len(vf) - runs[-1][1]) * hop_s)
    else:
        out["last_voiced_dur"] = np.nan
        out["final_leng_ratio"] = np.nan
        out["trail_unvoiced_s"] = np.nan

    # terminal slowing: syllable-rate (ZCR-peak) in last 0.6s vs last 1.5s
    def _rate(sec):
        t = _tail(ac, sr, sec)
        if len(t) < sr // 10:
            return np.nan
        z = librosa.feature.zero_crossing_rate(
            t, frame_length=int(sr * 0.025), hop_length=int(sr * 0.010))[0]
        above = z > z.mean()
        return float(np.sum(np.diff(above.astype(int)) > 0)) / sec
    r06, r15 = _rate(0.6), _rate(1.5)
    out["rate_recent"] = r06
    out["rate_decel"] = float(r06 / (r15 + 1e-6)) if (r06 == r06 and r15 == r15) else np.nan
    return out


def _spectral_feats(ac, sr) -> dict:
    """Spectral tilt / centroid trailing-off + articulatory settling."""
    tail = _tail(ac, sr, 1.5)
    out = {}
    keys = ["mfcc_delta_tail", "mfcc1_slope", "centroid_slope", "centroid_final_norm",
            "rolloff_final_norm", "flatness_tail", "spectral_bw_tail", "spectral_flux_tail"]
    if len(tail) < sr // 10:
        for k in keys:
            out[k] = np.nan
        return out
    hop = int(sr * 0.010); n_fft = int(sr * 0.025)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(y=tail, sr=sr, n_mfcc=13, hop_length=hop, n_fft=n_fft)
        mfcc_d = librosa.feature.delta(mfcc)
        cent = librosa.feature.spectral_centroid(y=tail, sr=sr, hop_length=hop, n_fft=n_fft)[0]
        roll = librosa.feature.spectral_rolloff(y=tail, sr=sr, hop_length=hop, n_fft=n_fft, roll_percent=0.85)[0]
        flat = librosa.feature.spectral_flatness(y=tail, hop_length=hop, n_fft=n_fft)[0]

    # articulation winding down: small MFCC-delta magnitude at the very end
    out["mfcc_delta_tail"] = float(np.abs(mfcc_d[:, -8:]).mean())
    out["mfcc1_slope"] = _slope(mfcc[1, -min(15, mfcc.shape[1]):]) if mfcc.shape[1] >= 5 else np.nan
    # centroid & rolloff normalized to their own tail distribution
    def _fin_norm(v):
        if len(v) < 5:
            return np.nan
        med = np.median(v); rng = max(np.percentile(v, 90) - np.percentile(v, 10), 1e-6)
        return float((v[-5:].mean() - med) / rng)
    out["centroid_slope"] = _slope(cent[-min(15, len(cent)):]) / (np.median(cent) + 1e-6) if len(cent) >= 3 else np.nan
    out["centroid_final_norm"] = _fin_norm(cent)
    out["rolloff_final_norm"] = _fin_norm(roll)
    out["flatness_tail"] = float(flat[-8:].mean()) if len(flat) >= 8 else float(flat.mean())
    # spectral bandwidth (voice quality breadth) and flux (articulatory activity)
    bw = librosa.feature.spectral_bandwidth(y=tail, sr=sr, hop_length=hop, n_fft=n_fft)[0]
    out["spectral_bw_tail"] = float(bw[-5:].mean() / (np.median(bw) + 1e-6)) if len(bw) >= 5 else np.nan
    S = np.abs(librosa.stft(tail, n_fft=n_fft, hop_length=hop))
    flux = (np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
            if S.shape[1] > 1 else np.array([0.0]))
    out["spectral_flux_tail"] = float(flux[-8:].mean()) if len(flux) >= 8 else float(flux.mean())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Discourse context
# ─────────────────────────────────────────────────────────────────────────────

def _discourse_feats(pause_start: float, pause_index: int,
                     prev_pause_durs: list = None) -> dict:
    """Cumulative silence/speech context — strictly causal (only prior pause durations).

    Causality: prev_pause_durs contains durations of pauses with pause_index < current,
    which all ended before the current pause began.
    """
    out = {}
    cum_sil = float(sum(prev_pause_durs)) if prev_pause_durs else 0.0
    cum_sp = max(0.0, pause_start - cum_sil)
    out["cum_sil"] = cum_sil
    out["cum_sp"] = cum_sp
    out["cum_sp_frac"] = float(cum_sp / (pause_start + 1e-6)) if pause_start > 0 else 1.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # pitch (9)
    "f0_voiced_frac", "f0_voiced_frac_near",
    "f0_final_norm", "f0_final_pctile", "f0_final_vs_med_semitone",
    "f0_slope_final", "f0_range_norm", "f0_slope_300ms", "f0_reset_semitone",
    # energy (10)
    "e_std_norm", "e_final_norm", "e_final_pctile",
    "e_slope_last10", "e_slope_last25", "e_slope_last50",
    "e_ratio_tail", "e_frac_low_tail", "e_total_drop", "e_full_pct_drop",
    # timing (9)
    "turn_dur", "log_turn_dur", "pause_index", "pause_rate",
    "last_voiced_dur", "final_leng_ratio", "trail_unvoiced_s",
    "rate_recent", "rate_decel",
    # spectral (8)
    "mfcc_delta_tail", "mfcc1_slope", "centroid_slope",
    "centroid_final_norm", "rolloff_final_norm", "flatness_tail",
    "spectral_bw_tail", "spectral_flux_tail",
    # discourse (3) — cumulative silence/speech context (causal)
    "cum_sil", "cum_sp", "cum_sp_frac",
]
N_FEATURES = len(FEATURE_NAMES)


def extract_features(x: np.ndarray, sr: int, pause_start: float,
                     pause_index: int, n_pauses_so_far: int = None,
                     prev_pause_durs: list = None) -> np.ndarray:
    """Feature vector for one pause. NaN allowed (tree models handle natively).

    CAUSALITY: audio is sliced to [0, pause_start) here, once, before anything.
    prev_pause_durs: list of durations of all pauses with pause_index < current
                     (used for cumulative discourse features; None → zeros).
    """
    ac = causal_slice(x, sr, pause_start)

    # ── shared computations (once) ───────────────────────────────────────────
    rms = _rms(ac, sr)
    pitch_win = _tail(ac, sr, PITCH_WIN_S)   # pyin only on turn-final window
    f0, vf = _pyin(pitch_win, sr)

    feat = {}
    feat.update(_pitch_feats(pitch_win, sr, f0, vf))
    feat.update(_energy_feats(ac, sr, rms))
    feat.update(_timing_feats(ac, sr, pause_start, pause_index, vf))
    feat.update(_spectral_feats(ac, sr))
    feat.update(_discourse_feats(pause_start, pause_index, prev_pause_durs))

    vec = np.array([feat[k] for k in FEATURE_NAMES], dtype=np.float32)
    assert vec.shape == (N_FEATURES,), f"shape {vec.shape} != {N_FEATURES}"
    return vec
