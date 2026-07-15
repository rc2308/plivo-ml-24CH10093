# NOTES — where the human turned the game

> The model in this repo works *because of human intervention at the decision
> points*, not because of the AI's first draft. The AI produced runnable code
> quickly; almost every choice that actually moved the score came from the human
> catching a mistake, questioning a number, or supplying the linguistic
> intuition. The turning points are logged below in order.

1. **The AI's first feature set used absolute pitch/energy (raw Hz, raw dB).**
   The human immediately rejected it: *"raw Hz will not transfer — a Hindi
   speaker's pitch range is different; normalise everything to the speaker's own
   turn."* This forced the switch to **relative features** — semitone ratios,
   percentile ranks, and slopes normalised to each turn's own range. Without this
   the model would have collapsed on the (mostly-Hindi) hidden test.

2. **The AI scored the model on English *in-sample* and reported 760 ms / AUC
   0.920 as if it were done.** The human flagged this as dishonest: *"that's the
   training set — of course it's 0.920. Show me an honest number."* The human
   mandated **GroupKFold grouped by `turn_id`** and, crucially, that CV be scored
   with the **actual delay metric from `score.py`**, not accuracy or AUC. The
   honest English number turned out to be ~1250 ms, not 760 ms — the AI's headline
   was pure optimism.

3. **The human re-read the brief — "hidden test is mostly Hindi" — and
   overrode the AI's English-only plan**, insisting on a **cross-lingual
   `train-EN → test-HI` proxy** as the single most important score (`_eval.py`).
   This is what exposed that the AI's fancy `pyin` pitch features had high
   *within-language* AUC but transferred to Hindi at barely-chance — a failure the
   AI never would have noticed on English alone.

4. **The human noticed the Hindi operating point was `threshold = 0.05`** and
   called it out: *"a 0.05 threshold means the model is firing on everything and
   just riding the silence timer — the scores aren't separated."* That diagnosis
   led to probability **calibration** and to trusting the honest Hindi ~850 ms
   over the flattering English 760 ms.

5. **Final-syllable lengthening and terminal pitch fall were the human's
   phonetic intuitions** ("statements drop in pitch and stretch the last
   syllable; a mid-turn hold does not"). The AI only implemented `final_leng_ratio`
   and the terminal F0 slopes *after* the human articulated the reasoning.

6. **The human insisted on leakage checks before trusting anything** — label
   shuffle, window-slicing causality (features must end exactly at `pause_start`,
   never see `pause_end`), GroupKFold turn-overlap, and a cold Hindi holdout. The
   human reviewed the grep output for `pause_end` line by line.

7. **The game-turning moment:** the AI's pipeline (absolute features + random
   split + in-sample scoring) *looked* excellent — 760 ms, AUC 0.92. The human
   replaced it with normalised features + turn-grouped CV + honest cross-lingual
   scoring, which revealed the true picture (~850 ms on Hindi) and prevented
   shipping a model that would have failed the real hidden test. The honest
   number is lower, but it is *real* — and that judgement call was entirely the
   human's.

**Key files**
- `features.py` — feature extraction (all relative/normalised, strictly causal).
- `train.py` — GroupKFold CV scored on the real delay metric; EN+HI training.
- `_eval.py` / `_exp.py` / `_subset.py` / `_diag.py` — the human's honest
  evaluation harness (cross-lingual + per-feature transfer diagnostics).
- `leak_tests.py` — the five integrity checks.
- `score.py` — the official scorer (unchanged).
