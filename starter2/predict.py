"""EOT predictor — loads saved model.pkl, never refits.

Usage:
    python predict.py --data_dir ../eot_handout/eot_data/english --out predictions.csv
    python predict.py --data_dir ../eot_handout/eot_data/hindi   --out predictions_hindi.csv
"""
import argparse
import csv
import os
import pickle
import warnings

import numpy as np

from features import load_wav, extract_features

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="model.pkl")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    print(f"Loaded model: {artifact['model_name']} from {args.model}")

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))

    # Pre-build turn_pauses dict for cumulative discourse features (causal)
    turn_pauses = {}
    for r in rows:
        turn_pauses.setdefault(r["turn_id"], []).append(r)

    wav_cache = {}
    results = []

    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in wav_cache:
            wav_cache[path] = load_wav(path)
        x, sr = wav_cache[path]

        pause_index = int(r["pause_index"])
        tid = r["turn_id"]

        # Prior pause durations only (pause_index < current) — causal
        prev_durs = [
            float(prev["pause_end"]) - float(prev["pause_start"])
            for prev in turn_pauses[tid]
            if int(prev["pause_index"]) < pause_index
        ]

        feat = extract_features(x, sr, float(r["pause_start"]),
                                pause_index, pause_index + 1,
                                prev_pause_durs=prev_durs)
        p_eot = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
        results.append({
            "turn_id":    r["turn_id"],
            "pause_index": r["pause_index"],
            "p_eot":      f"{p_eot:.4f}",
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        w.writeheader()
        w.writerows(results)

    print(f"Wrote {len(results)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
