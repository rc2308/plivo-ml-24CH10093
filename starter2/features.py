"""EOT feature extraction — strict causal, cross-lingual.

CAUSALITY CONTRACT
    extract_features() slices x[0 : int(pause_start*sr)] ONCE at the top.
    No helper ever sees audio at/after pause_start.

DESIGN
    Every feature is a slope, ratio, or turn-normalised value.
    Raw Hz/dB would not transfer across speakers or languages.
    pyin is run once on the last PITCH_WIN_S seconds of causal audio.
"""
import warnings

import librosa
import numpy as np
import soundfile as sf

SR_HOP_MS   = 10
FRAME_MS    = 25
PITCH_WIN_S = 2.5
FMIN, FMAX  = 60.0, 400.0


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_wav(path: str):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def causal_slice(x, sr, pause_start):
    end = min(int(round(pause_start * sr)), len(x))
    return x[:end]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _tail(a, sr, sec):
    return a[max(0, len(a) - int(sec * sr)):]


def _rms(a, sr):
    fl = int(sr * FRAME_MS / 1000)
    hp = int(sr * SR_HOP_MS / 1000)
    if len(a) < fl:
        return np.array([np.sqrt(np.mean(a**2) + 1e-12)], np.float32)
    n   = 1 + (len(a) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return np.sqrt(np.mean(a[idx]**2, axis=1) + 1e-12).astype(np.float32)


def _pyin(a, sr):
    hop = int(sr * SR_HOP_MS / 1000)
    if len(a) < hop * 4:
        return np.array([np.nan], np.float32), np.array([False])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, vflag, _ = librosa.pyin(a.astype(np.float32), fmin=FMIN, fmax=FMAX,
                                    sr=sr, hop_length=hop, fill_na=np.nan)
    return f0.astype(np.float32), vflag.astype(bool)


def _voiced_runs(vf):
    segs, inv, start = [], False, 0
    for i, v in enumerate(vf):
        if v and not inv:
            start, inv = i, True
        elif not v and inv:
            segs.append((start, i)); inv = False
    if inv:
        segs.append((start, len(vf)))
    return segs


def _slope(v):
    valid = v[~np.isnan(v)]
    if len(valid) < 3:
        return np.nan
    idx = np.where(~np.isnan(v))[0].astype(float)
    s, _ = np.polyfit(idx, valid, 1)
    return float(s)


def _pct_rank(value, arr):
    a = arr[~np.isnan(arr)]
    if len(a) < 3 or np.isnan(value):
        return np.nan
    return float((a < value).mean())


def _nanmedian_safe(v, fallback=1.0):
    m = np.nanmedian(v)
    return float(m) if np.isfinite(m) else fallback


# ── Feature families ──────────────────────────────────────────────────────────

def _pitch_feats(ac, sr, f0, vf):
    out = {}
    voiced = f0[~np.isnan(f0)]

    out["f0_voiced_frac"]      = float(vf.mean()) if len(vf) else np.nan
    near = vf[-15:] if len(vf) >= 15 else vf
    out["f0_voiced_frac_near"] = float(near.mean()) if len(near) else np.nan

    if len(voiced) >= 4:
        med    = float(np.median(voiced))
        p10, p90 = np.percentile(voiced, [10, 90])
        rng    = max(p90 - p10, 1e-6)
        last   = voiced[-min(5, len(voiced)):]
        last_m = float(last.mean())
        out["f0_final_norm"]           = (last_m - med) / rng
        out["f0_final_pctile"]         = _pct_rank(last_m, voiced)
        out["f0_final_vs_med_semitone"] = 12.0 * np.log2((last_m + 1e-6) / (med + 1e-6))
        n = min(12, len(voiced))
        out["f0_slope_final"]          = _slope(voiced[-n:]) / (med + 1e-6)
        out["f0_range_norm"]           = rng / (med + 1e-6)
        # last 10 vs prior voiced: continuation rise flag
        last10 = voiced[-10:] if len(voiced) >= 10 else voiced
        first_half = voiced[:max(1, len(voiced)//2)]
        out["f0_final_vs_turn_early"]  = (last10.mean() - first_half.mean()) / (med + 1e-6)
    else:
        for k in ["f0_final_norm", "f0_final_pctile", "f0_final_vs_med_semitone",
                  "f0_slope_final", "f0_range_norm", "f0_final_vs_turn_early"]:
            out[k] = np.nan

    # slope over last 300 ms
    f0_t, _ = _pyin(_tail(ac, sr, 0.3), sr)
    med_all  = _nanmedian_safe(f0)
    out["f0_slope_300ms"] = _slope(f0_t) / (med_all + 1e-6)

    # pitch reset between last two voiced runs
    runs = _voiced_runs(vf)
    if len(runs) >= 2:
        (s1, e1), (s2, e2) = runs[-1], runs[-2]
        a = f0[s1:e1]; b = f0[s2:e2]
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        out["f0_reset_semitone"] = (12.0 * np.log2((a.mean() + 1e-6) / (b.mean() + 1e-6))
                                    if len(a) and len(b) else np.nan)
    else:
        out["f0_reset_semitone"] = np.nan

    # pitch declination: linear trend over entire voiced sequence
    if len(voiced) >= 6:
        out["f0_declination"] = _slope(voiced) / (_nanmedian_safe(voiced) + 1e-6)
    else:
        out["f0_declination"] = np.nan

    # voiced fraction in last 500ms vs full window (terminal devoicing)
    vf_tail = vf[-50:] if len(vf) >= 50 else vf
    full_frac = float(vf.mean()) if len(vf) else np.nan
    tail_frac = float(vf_tail.mean()) if len(vf_tail) else np.nan
    out["f0_devoice_ratio"] = (tail_frac / (full_frac + 1e-6)
                               if full_frac == full_frac and full_frac > 0 else np.nan)

    return out


def _energy_feats(ac, sr, rms):
    out = {}
    db  = 20 * np.log10(rms + 1e-12)
    lo, hi = np.percentile(db, [5, 95])
    rng = max(hi - lo, 1e-6)
    med = float(np.median(db))

    out["e_std_norm"]      = float(db.std() / rng)
    tailn = db[-8:] if len(db) >= 8 else db
    out["e_final_norm"]    = float((tailn.mean() - med) / rng)
    out["e_final_pctile"]  = _pct_rank(float(tailn.mean()), db)
    out["e_slope_last10"]  = (_slope(db[-10:]) / rng) if len(db) >= 10 else np.nan
    out["e_slope_last25"]  = (_slope(db[-25:]) / rng) if len(db) >= 25 else np.nan
    n50 = max(1, int(500 / SR_HOP_MS))
    tail50 = db[-n50:] if len(db) >= n50 else db
    out["e_slope_last50"]  = (_slope(tail50) / rng) if len(tail50) >= 3 else np.nan
    out["e_ratio_tail"]    = float(rms[-5:].mean() / (rms.mean() + 1e-12)) if len(rms) >= 5 else np.nan
    gate = lo + 0.15 * rng
    last30 = db[-30:] if len(db) >= 30 else db
    out["e_frac_low_tail"] = float((last30 < gate).mean())
    out["e_total_drop"]    = float((db[-1] - db[0]) / rng) if len(db) >= 2 else np.nan
    if len(db) >= n50:
        out["e_full_pct_drop"] = float(
            np.percentile(db[-n50:], 25) - np.percentile(db, 25)) / rng
    else:
        out["e_full_pct_drop"] = np.nan

    # energy in last 1s vs prior 1s (relative decay)
    n100 = int(1.0 * sr)
    n200 = int(2.0 * sr)
    if len(ac) >= n200:
        e_last1 = float(np.mean(ac[-n100:]**2) + 1e-12)
        e_prev1 = float(np.mean(ac[-n200:-n100]**2) + 1e-12)
        out["e_last1_vs_prev1"] = float(np.log(e_last1 / e_prev1))
    else:
        out["e_last1_vs_prev1"] = np.nan

    return out


def _timing_feats(ac, sr, pause_start, pause_index, vf):
    out = {}
    out["turn_dur"]     = float(pause_start)
    out["log_turn_dur"] = float(np.log1p(pause_start))
    out["pause_index"]  = float(pause_index)
    out["pause_rate"]   = float(pause_index) / (float(pause_start) / 60.0 + 1e-6)

    runs   = _voiced_runs(vf)
    hop_s  = SR_HOP_MS / 1000.0
    if runs:
        durs = np.array([(e - s) * hop_s for s, e in runs])
        out["last_voiced_dur"]  = float(durs[-1])
        out["final_leng_ratio"] = (float(durs[-1] / (np.median(durs[:-1]) + 1e-6))
                                   if len(durs) >= 2 else np.nan)
        out["trail_unvoiced_s"] = float((len(vf) - runs[-1][1]) * hop_s)
        # number of voiced segments (syllable-count proxy)
        out["n_voiced_segs"]    = float(len(runs))
        out["mean_voiced_dur"]  = float(durs.mean())
    else:
        for k in ["last_voiced_dur", "final_leng_ratio", "trail_unvoiced_s",
                  "n_voiced_segs", "mean_voiced_dur"]:
            out[k] = np.nan

    # rate (ZCR-peak) in 0.6s vs 1.5s
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
    out["rate_decel"]  = (float(r06 / (r15 + 1e-6))
                          if (r06 == r06 and r15 == r15) else np.nan)

    # rate 1.0s vs 2.0s (broader deceleration)
    r10, r20 = _rate(1.0), _rate(2.0)
    out["rate_decel_broad"] = (float(r10 / (r20 + 1e-6))
                               if (r10 == r10 and r20 == r20) else np.nan)

    return out


def _spectral_feats(ac, sr):
    tail = _tail(ac, sr, 1.5)
    out  = {}
    keys = ["mfcc_delta_tail", "mfcc1_slope", "centroid_slope", "centroid_final_norm",
            "rolloff_final_norm", "flatness_tail", "spectral_bw_tail", "spectral_flux_tail",
            "mfcc_mean1", "mfcc_mean2", "mfcc_mean3", "mfcc_mean4",
            "mfcc_std1",  "mfcc_std2",  "mfcc_std3",
            "mfcc_delta_mean1", "mfcc_delta_mean2", "mfcc_delta_mean3"]
    if len(tail) < sr // 10:
        for k in keys:
            out[k] = np.nan
        return out

    hop   = int(sr * 0.010)
    n_fft = int(sr * 0.025)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc   = librosa.feature.mfcc(y=tail, sr=sr, n_mfcc=13, hop_length=hop, n_fft=n_fft)
        mfcc_d = librosa.feature.delta(mfcc)
        cent   = librosa.feature.spectral_centroid(y=tail, sr=sr, hop_length=hop, n_fft=n_fft)[0]
        roll   = librosa.feature.spectral_rolloff(y=tail, sr=sr, hop_length=hop, n_fft=n_fft,
                                                   roll_percent=0.85)[0]
        flat   = librosa.feature.spectral_flatness(y=tail, hop_length=hop, n_fft=n_fft)[0]

    out["mfcc_delta_tail"] = float(np.abs(mfcc_d[:, -8:]).mean())
    out["mfcc1_slope"]     = (_slope(mfcc[1, -min(15, mfcc.shape[1]):]
                               ) if mfcc.shape[1] >= 5 else np.nan)

    def _fin_norm(v):
        if len(v) < 5:
            return np.nan
        med = np.median(v)
        rng = max(np.percentile(v, 90) - np.percentile(v, 10), 1e-6)
        return float((v[-5:].mean() - med) / rng)

    out["centroid_slope"]      = (_slope(cent[-min(15, len(cent)):]) / (np.median(cent) + 1e-6)
                                  if len(cent) >= 3 else np.nan)
    out["centroid_final_norm"] = _fin_norm(cent)
    out["rolloff_final_norm"]  = _fin_norm(roll)
    out["flatness_tail"]       = float(flat[-8:].mean()) if len(flat) >= 8 else float(flat.mean())

    bw   = librosa.feature.spectral_bandwidth(y=tail, sr=sr, hop_length=hop, n_fft=n_fft)[0]
    out["spectral_bw_tail"]    = float(bw[-5:].mean() / (np.median(bw) + 1e-6)) if len(bw) >= 5 else np.nan
    S    = np.abs(librosa.stft(tail, n_fft=n_fft, hop_length=hop))
    flux = (np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)) if S.shape[1] > 1 else np.array([0.0]))
    out["spectral_flux_tail"]  = float(flux[-8:].mean()) if len(flux) >= 8 else float(flux.mean())

    # MFCC statistics over final window (language-agnostic articulatory posture)
    tail_cols = mfcc[:, -min(20, mfcc.shape[1]):]
    d_tail    = mfcc_d[:, -min(20, mfcc_d.shape[1]):]
    for i in range(1, 5):
        out[f"mfcc_mean{i}"] = float(tail_cols[i].mean()) if tail_cols.shape[1] > 0 else np.nan
    for i in range(1, 4):
        out[f"mfcc_std{i}"]  = float(tail_cols[i].std())  if tail_cols.shape[1] > 0 else np.nan
    for i in range(1, 4):
        out[f"mfcc_delta_mean{i}"] = float(d_tail[i].mean()) if d_tail.shape[1] > 0 else np.nan

    return out


def _discourse_feats(pause_start, pause_index, prev_pause_durs=None):
    out     = {}
    cum_sil = float(sum(prev_pause_durs)) if prev_pause_durs else 0.0
    cum_sp  = max(0.0, pause_start - cum_sil)
    out["cum_sil"]      = cum_sil
    out["cum_sp"]       = cum_sp
    out["cum_sp_frac"]  = float(cum_sp / (pause_start + 1e-6)) if pause_start > 0 else 1.0
    # silence-to-speech ratio in turn so far
    out["sil_sp_ratio"] = cum_sil / (cum_sp + 1e-6)
    # relative position: what fraction of this turn's elapsed time has been silence
    out["sil_frac_turn"] = cum_sil / (pause_start + 1e-6)
    # log pause index (diminishing returns of additional pauses)
    out["log_pause_idx"] = float(np.log1p(pause_index))
    # mean previous pause duration (longer avg pauses → more likely mid-turn)
    if prev_pause_durs:
        out["mean_prev_dur"] = float(np.mean(prev_pause_durs))
        out["max_prev_dur"]  = float(np.max(prev_pause_durs))
    else:
        out["mean_prev_dur"] = 0.0
        out["max_prev_dur"]  = 0.0
    return out


# ── Public API ────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # pitch (11)
    "f0_voiced_frac", "f0_voiced_frac_near",
    "f0_final_norm", "f0_final_pctile", "f0_final_vs_med_semitone",
    "f0_slope_final", "f0_range_norm", "f0_slope_300ms", "f0_reset_semitone",
    "f0_declination", "f0_devoice_ratio", "f0_final_vs_turn_early",
    # energy (11)
    "e_std_norm", "e_final_norm", "e_final_pctile",
    "e_slope_last10", "e_slope_last25", "e_slope_last50",
    "e_ratio_tail", "e_frac_low_tail", "e_total_drop", "e_full_pct_drop",
    "e_last1_vs_prev1",
    # timing (11)
    "turn_dur", "log_turn_dur", "pause_index", "pause_rate",
    "last_voiced_dur", "final_leng_ratio", "trail_unvoiced_s",
    "n_voiced_segs", "mean_voiced_dur",
    "rate_recent", "rate_decel", "rate_decel_broad",
    # spectral (18)
    "mfcc_delta_tail", "mfcc1_slope", "centroid_slope",
    "centroid_final_norm", "rolloff_final_norm", "flatness_tail",
    "spectral_bw_tail", "spectral_flux_tail",
    "mfcc_mean1", "mfcc_mean2", "mfcc_mean3", "mfcc_mean4",
    "mfcc_std1",  "mfcc_std2",  "mfcc_std3",
    "mfcc_delta_mean1", "mfcc_delta_mean2", "mfcc_delta_mean3",
    # discourse (8)
    "cum_sil", "cum_sp", "cum_sp_frac",
    "sil_sp_ratio", "sil_frac_turn", "log_pause_idx",
    "mean_prev_dur", "max_prev_dur",
]
N_FEATURES = len(FEATURE_NAMES)


def extract_features(x, sr, pause_start, pause_index, n_pauses_so_far=None,
                     prev_pause_durs=None):
    """Feature vector for one pause. NaN allowed (tree models handle natively).

    CAUSALITY: audio is sliced to [0, pause_start) here, once, before anything.
    """
    ac          = causal_slice(x, sr, pause_start)
    rms         = _rms(ac, sr)
    pitch_win   = _tail(ac, sr, PITCH_WIN_S)
    f0, vf      = _pyin(pitch_win, sr)

    feat = {}
    feat.update(_pitch_feats(pitch_win, sr, f0, vf))
    feat.update(_energy_feats(ac, sr, rms))
    feat.update(_timing_feats(ac, sr, pause_start, pause_index, vf))
    feat.update(_spectral_feats(ac, sr))
    feat.update(_discourse_feats(pause_start, pause_index, prev_pause_durs))

    vec = np.array([feat[k] for k in FEATURE_NAMES], dtype=np.float32)
    assert vec.shape == (N_FEATURES,), f"shape {vec.shape} != {N_FEATURES}"
    return vec
