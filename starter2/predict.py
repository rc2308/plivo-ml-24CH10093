"""EOT predictor — loads saved model.pkl, never refits.

Usage:
    python predict.py --data_dir ../eot_data/english --out predictions_english.csv
    python predict.py --data_dir ../eot_data/hindi   --out predictions_hindi.csv
"""
import argparse
import csv
import pickle
import warnings
from pathlib import Path

import numpy as np

from features import load_wav, extract_features

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out",   default="predictions.csv")
    ap.add_argument("--model", default="model.pkl")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    with open(args.model, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    print(f"Loaded: {artifact['model_name']} from {args.model}")

    rows = list(csv.DictReader(open(data_dir / "labels.csv")))
    turn_pauses = {}
    for r in rows:
        turn_pauses.setdefault(r["turn_id"], []).append(r)

    wav_cache, results = {}, []
    for r in rows:
        path = str(data_dir / r["audio_file"])
        if path not in wav_cache:
            wav_cache[path] = load_wav(path)
        x, sr = wav_cache[path]
        pi  = int(r["pause_index"])
        tid = r["turn_id"]
        prev_durs = [float(p["pause_end"]) - float(p["pause_start"])
                     for p in turn_pauses[tid] if int(p["pause_index"]) < pi]
        feat  = extract_features(x, sr, float(r["pause_start"]), pi, pi + 1, prev_durs)
        p_eot = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
        results.append({"turn_id": r["turn_id"], "pause_index": r["pause_index"],
                        "p_eot": f"{p_eot:.4f}"})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        w.writeheader(); w.writerows(results)
    print(f"Wrote {len(results)} predictions → {args.out}")


if __name__ == "__main__":
    main()
