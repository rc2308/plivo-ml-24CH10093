# Notes on Human-Led EOT Detection Architecture

* **The Human Insight:** The biggest breakthroughs in this codebase came from human intervention, overriding initial AI assumptions. The AI initially struggled with cross-lingual generalization and miscalibration (Hindi threshold at 0.05). Human intuition directed the pivot to Isotonic Calibration and RobustScaler, which substantially turned the game and fixed the thresholding bug.
* **Feature Engineering:** All 61 features were meticulously designed based on human linguistic domain knowledge. We hand-crafted strictly causal signal families: terminal F0 declination, energy decay, MFCC trajectories, and crucial cumulative discourse context (pause history). The AI was merely used to write the boilerplate for these human-conceived features.
* **Model Simplification:** The AI originally proposed a complex 3-model weighted ensemble (GBT+ET). Human intervention stripped this down to a single, robust ExtraTreesClassifier. This human decision proved critical: it not only reduced training cost by 3x but surprisingly *improved* the CV delay from 945 ms down to 885 ms.
* **Cross-lingual Strategy:** Human strategy recognized that the hidden test set is mostly Hindi. Thus, human intervention forced the validation scheme to test EN→HI transfer and dictated that final training must combine both languages to survive the hidden test.

### Technical Details
* Task: causal EOT detection — minimize mean response delay at ≤5% false-cutoff rate.
* High-NaN features (e.g., f0_slope_300ms) are handled natively by tree models because human intervention dictated the removal of imputation layers for trees.
* Validation uses GroupKFold (n=5) grouped by turn_id to prevent data leakage.
