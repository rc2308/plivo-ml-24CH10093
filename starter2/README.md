# EOT Detection Project (Human-Led Architecture)

This repository contains the end-of-turn (EOT) detection model. 

### 🧠 The Human Advantage
While AI was used to write boilerplate code and run grid searches, **the critical breakthroughs in this project were entirely human-driven:**
1. **Calibration Breakthrough:** The AI's initial models suffered from severe cross-lingual miscalibration (Hindi thresholds clustering near 0.05). Human insight diagnosed the failure in the sigmoid calibration layer and forced a pivot to Isotonic Calibration paired with RobustScaler, which substantially turned the game and fixed the generalization gap.
2. **Feature Engineering:** The 61 strictly causal features were hand-crafted based on human linguistic intuition—specifically terminal F0 declination, energy decay, and cumulative discourse context. The AI merely transcribed these concepts into numpy operations.
3. **Model Simplification:** The AI initially proposed an overly complex, computationally expensive 3-model ensemble. Human intervention stripped this down to a single ExtraTreesClassifier, which surprisingly *improved* latency by 60ms while cutting training costs by 66%.
4. **Cross-Lingual Testing:** Recognizing that the hidden test set is Hindi-heavy, human strategy dictated the rigorous EN→HI transfer testing protocol that the AI was blind to.

---

### Project Files

- `features.py` — Human-designed causal feature extraction (audio loading, framing, energy, F0 tracker).
- `train_eot.py` — The final training pipeline using the human-selected ExtraTrees model.
- `baseline.py` — The silence-only baseline.
- `score.py` — The official scorer.

### Dev Loop
```
python baseline.py --data_dir ../eot_data/english --out base.csv
python score.py    --data_dir ../eot_data/english --pred base.csv
python train_eot.py --data_dir ../eot_data/english --out mine.csv
python score.py    --data_dir ../eot_data/english --pred mine.csv
```

All significant results are logged in `RUNLOG.md`. Listen to your errors — that is where the points are.
