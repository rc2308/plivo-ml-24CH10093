# Notes

* Task: causal EOT detection — output p_eot per pause, minimize mean response delay at ≤5% false-cutoff rate.
* All features are strictly causal: only audio up to `pause_start` is used; `pause_end`, future silence, and future speech are never accessed.
* Every feature is turn-normalized (slope, ratio, percentile rank) so raw Hz/dB differences across speakers and languages do not hurt generalization.
* Key signal families: terminal F0 declination + devoicing, energy decay in final 1s, MFCC trajectory, speaking-rate deceleration, and cumulative discourse context (pause history).
* Model: isotonic-calibrated Ensemble of GradientBoostingClassifier (33%) and ExtraTreesClassifier (67%), selected by GroupKFold CV grouped by turn_id.
* Validation uses GroupKFold (n=5) grouped by turn_id so no pause from the same turn leaks across folds.
* CV delay improved from 1600 ms (baseline) → 1016 ms (Run 3 HGBT) → 945 ms (Run 4 Ensemble), a 41% reduction.
* Cross-lingual gap: EN→HI 850 ms vs HI→EN 1445 ms; Hindi-as-test-only performs better because training on both languages together helps EN→HI generalization.
* High-NaN features (f0_slope_300ms 23%, f0_reset_semitone 9%) handled natively by tree models via isotonic calibration pipeline with SimpleImputer fallback for LR.
* Hidden test is expected to be mostly Hindi; training on combined EN+HI data is therefore critical.
