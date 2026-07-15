# Run Log

## Run 1 — Baseline (silence-only)
```
python baseline.py --data_dir eot_handout/eot_data/english --out predictions.csv
python score.py    --data_dir eot_handout/eot_data/english --pred predictions.csv
```
- English: AUC=0.506, delay=**1600 ms**, interrupted=0.0%
- Notes: Always predicts p_eot=1.0. Scorer uses 1.6s timeout. This is the floor.

---

## Run 2 — GBT v2 (22 features, pyin F0)
```
python train.py --data_dir eot_handout/eot_data/english --out predictions.csv --clf gbt
```
- CV AUC: **0.629 ± 0.078** (5-fold grouped by turn)
- English train AUC=1.000 (overfit), delay=100ms (not honest)
- Notes: Cross-lingual gap: EN→HI 850 ms vs HI→EN 1445 ms; combined training helps EN→HI generalization.

---

## Run 5 — Single ExtraTrees v3 (61 features, no ensemble)
```
python train.py --data_dirs ../eot_data/english ../eot_data/hindi
python predict.py --data_dir ../eot_data/english --out predictions_english.csv
python predict.py --data_dir ../eot_data/hindi   --out predictions_hindi.csv
python score.py --data_dir ../eot_data/english --pred predictions_english.csv
python score.py --data_dir ../eot_data/hindi   --pred predictions_hindi.csv
```

### CV Results (GroupKFold, 5 splits, @≤5% FC):
| Model | CV delay | ± ms |
|-------|----------|------|
| **ExtraTrees** | **885 ms** | 115 |
| GBT | 967 ms | 97 |
| LR | 1257 ms | 214 |

**Selected: ExtraTreesClassifier (n=500, max_features=0.55, isotonic calibration)**

### On-train scores (overfit — reference only):
- English: AUC=1.000, delay=**100 ms**, interrupted=4.0%, threshold=0.30
- Hindi:   AUC=1.000, delay=**100 ms**, interrupted=3.0%, threshold=0.30

### Cross-lingual:
- EN→HI: 850 ms, cutoff=5.0%, thresh=0.05
- HI→EN: 1487 ms, cutoff=5.0%, thresh=0.70

### Key finding:
Single ET (885 ms) **beats** the 3-model ensemble (945 ms) by 60 ms.
Ensemble adds 3× training cost with no benefit — dropped permanently.

### Full improvement history:
| Run | Model | CV Delay | vs Baseline |
|-----|-------|----------|-------------|
| 1 | Baseline | 1600 ms | — |
| 3 | HGBT, 39 feats | 1016 ms | −37% |
| 4 | Ensemble GBT+ET, 61 feats | 945 ms | −41% |
| **5** | **Single ET, 61 feats** | **885 ms** | **−45%** |

---

## Run 3 — HistGradientBoosting (22 features, pyin F0)
```
python train.py --data_dir eot_handout/eot_data/english --out predictions.csv
python score.py --data_dir eot_handout/eot_data/english --pred predictions.csv
python predict.py --data_dir eot_handout/eot_data/hindi --out predictions_hindi.csv
python score.py --data_dir eot_handout/eot_data/hindi --pred predictions_hindi.csv
```
- CV AUC: **0.594 ± 0.028** (stable)
- English: AUC=0.920, delay=**760 ms**, interrupted=5.0%, threshold=0.55, action_delay=450ms
- Hindi:   AUC=0.603, delay=**850 ms**, interrupted=5.0%, threshold=0.05, action_delay=850ms
- Top features: E4_pause_density(+0.122), D2_late_speaking_rate(+0.072), C2_final_syllable_len(+0.064)
- Notes: 2× better than baseline. Hindi generalizes reasonably (+90ms gap). Hindi threshold=0.05 suggests model scores are not calibrated — to fix.

---

## Run 4 — Improved v2: 61 features, Ensemble(GBT+ET) (current)
```
python train.py --data_dirs ../eot_data/english ../eot_data/hindi --out predictions.csv --model_out model.pkl
python predict.py --data_dir ../eot_data/english --out predictions_english.csv
python predict.py --data_dir ../eot_data/hindi   --out predictions_hindi.csv
python score.py --data_dir ../eot_data/english --pred predictions_english.csv
python score.py --data_dir ../eot_data/hindi   --pred predictions_hindi.csv
```

### CV Results (GroupKFold, 5 splits, English+Hindi combined, @≤5% FC):
| Model    | CV delay (ms) | ± ms |
|----------|--------------|------|
| Ensemble | **945**      | 0    |
| ET       | 957          | 97   |
| GBT      | 967          | 97   |
| RF       | 993          | 64   |
| HGBT     | 1016         | 81   |
| LR       | 1257         | 214  |
| Baseline | 1600         | —    |

**Selected: Ensemble (GBT 33% + ET 67%)**

### On-train scores (overfit, for reference only):
- English: AUC=1.000, delay=**100 ms**, interrupted=2.0%, threshold=0.35
- Hindi:   AUC=1.000, delay=**100 ms**, interrupted=4.0%, threshold=0.35

### Cross-lingual generalization (train one lang, test other):
| Direction | Delay  | Cutoff | Threshold |
|-----------|--------|--------|-----------|
| EN → HI   | 850 ms | 5.0%   | 0.05      |
| HI → EN   | 1445 ms| 5.0%   | 0.55      |

### New features added vs Run 3 (39 → 61 features):
- **Pitch**: `f0_declination`, `f0_devoice_ratio`, `f0_final_vs_turn_early`
- **Energy**: `e_last1_vs_prev1` (log ratio last 1s vs prior 1s)
- **Timing**: `n_voiced_segs`, `mean_voiced_dur`, `rate_decel_broad`
- **Spectral**: 10 MFCC stats (mean/std for coeffs 1-4, delta coeffs 1-3)
- **Discourse**: `sil_sp_ratio`, `sil_frac_turn`, `log_pause_idx`, `mean_prev_dur`, `max_prev_dur`

### Improvement over baseline:
- Baseline → Run 4: **1600 ms → 945 ms CV delay** (−655 ms, **−41%**)
- Run 3 → Run 4: HGBT 1016 ms → Ensemble 945 ms (−71 ms, **−7%** further improvement)
