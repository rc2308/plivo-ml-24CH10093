# Run Log — EOT Detection

## Run 1 — Baseline (silence-only, p_eot = 1)
```
python baseline.py --data_dir eot_handout/eot_data/english --out predictions.csv
python score.py    --data_dir eot_handout/eot_data/english --pred predictions.csv
```
- English: AUC = 0.506, delay = **1600 ms**, interrupted = 0.0%
- Pure silence timer. This is the floor. Must beat it.

---

## Run 2 — First ML attempt (AI draft: absolute features, random split)
- Features: absolute F0 / raw energy means; validation = random pause split.
- English **in-sample**: AUC = 0.920, delay = **760 ms** — *looked* great.
- **Human intervention:** "That is the training set scored on itself, and the
  split leaks whole turns across train/val. Not honest. Regroup by `turn_id`
  and score on the real delay metric." → Run 3.

---

## Run 3 — Honest evaluation (human-mandated): the discrepancy exposed
GroupKFold(5) by `turn_id`, scored with `score.py`'s delay metric.

| Metric | AI's in-sample claim | **Honest (OOF / cross-lingual)** |
|---|---|---|
| English AUC | 0.920 | **~0.63** |
| English delay | 760 ms | **~1250 ms** |
| Hindi delay (real target) | — | **~850 ms** |
| Hindi operating threshold | — | 0.05 ⚠ |

- **Discrepancy:** the 760 ms headline was in-sample optimism. The honest
  English OOF is ~1250 ms; the number that actually matters (Hindi) is ~850 ms.
- **Suspicious signal:** Hindi's best threshold is 0.05 — the model fires on
  nearly every pause and rides the silence timer, i.e. its probabilities are
  poorly separated. Fixed with calibration; treated ~850 ms as the true score.

---

## Run 4 — Cross-lingual transfer study (`_exp.py`, `_subset.py`, `_diag.py`)
Train EN → test HI (the honest proxy, since the hidden set is mostly Hindi):

| Feature family | within-HI AUC | **EN→HI transfer AUC** | verdict |
|---|---|---|---|
| `pyin` pitch (raw) | high (~0.74) | **~0.50 (chance)** | overfits language |
| normalised prosody (slopes/ratios/semitones) | ~0.70 | **~0.66** | transfers |
| discourse (pause index, cumulative silence) | — | strong | transfers |

- **Human conclusion:** keep only features that transfer. High within-language
  AUC is a trap; cross-lingual AUC is the truth for this task.

---

## Leakage / integrity checks (`python leak_tests.py`)
| Check | Expected | Result |
|---|---|---|
| 1. Label-shuffle OOF AUC | ≈ 0.5 | ✅ ≈ 0.5 (no structural leak) |
| 2. Feature window ends at `pause_start` | never past it | ✅ |
| 3. `pause_end` grep audit | scorer/prior-pause only | ✅ |
| 4. GroupKFold turn overlap | 0 | ✅ |
| 5. Cold Hindi holdout (20 unseen turns) | ≈ OOF range | ✅ honest ~850 ms |

---

## Run 5 — Red flags fixed (human-directed): revived features + LR+ET ensemble
Two bugs the human caught and made the AI fix (`ship.py`):
1. **Dead causal features.** `predict.py`/`train.py` never passed
   `prev_pause_durs`, so the cumulative-silence discourse features were always
   zero. Threaded them through everywhere → the features are now alive.
2. **Probabilities not separated** (the `threshold = 0.05` red flag). Replaced
   the single tree with an **LR + ExtraTrees soft-voting ensemble** trained on
   English + Hindi.

Honest GroupKFold(5) delay (EN+HI trained, scored per language):
| Set | Honest delay | Honest AUC | Operating threshold |
|---|---|---|---|
| English (OOF) | 1299 ms | 0.639 | 0.45 |
| **Hindi (OOF, real target)** | **831 ms** | **0.735** | **0.45 (was 0.05 ✅)** |

The threshold jumping from 0.05 → 0.45 is the proof the separation bug is fixed:
the model now discriminates instead of firing on everything and riding the timer.

---

## Final
| Set | Honest delay | Honest AUC | Note |
|---|---|---|---|
| English (OOF) | 1299 ms | 0.639 | vs 1600 ms baseline |
| **Hindi (OOF, real target)** | **831 ms** | **0.735** | **−48% vs baseline** |

Deployed model: **LR + ExtraTrees soft-vote**, trained on English + Hindi, all 39
features strictly causal (`prev_pause_durs` threaded). The reported number is the
honest out-of-fold one — the human's call to reject the flattering in-sample
760 ms, revive the dead features, and demand separated probabilities is what
makes this both trustworthy and better.
