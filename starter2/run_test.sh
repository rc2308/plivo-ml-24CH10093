#!/usr/bin/env bash
# run_test.sh — the full dev loop in one command
# Usage: bash run_test.sh [english|hindi|both]
#
# What it does:
#   1. Train on English data   → saves model.pkl + predictions.csv
#   2. Score English           → prints mean response delay
#   3. Run predict.py on Hindi → saves predictions_hindi.csv
#   4. Score Hindi             → prints mean response delay
#   5. Appends results to RUNLOG.md

set -e
PYTHON="/Users/rounak/speedrun/env/bin/python"
DATA="eot_handout/eot_data"
LANG="${1:-both}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "============================================"
echo "  EOT Test Run — $TIMESTAMP"
echo "============================================"

# ---------- TRAIN (always on English) ----------
echo ""
echo ">>> [1/4] Training on English..."
$PYTHON train.py \
    --data_dir "$DATA/english" \
    --out predictions.csv \
    --clf gbt

# ---------- SCORE ENGLISH ----------
echo ""
echo ">>> [2/4] Scoring English..."
EN_RESULT=$($PYTHON score.py --data_dir "$DATA/english" --pred predictions.csv)
echo "$EN_RESULT"

# ---------- PREDICT HINDI ----------
echo ""
echo ">>> [3/4] Generating Hindi predictions..."
$PYTHON predict.py \
    --data_dir "$DATA/hindi" \
    --out predictions_hindi.csv

# ---------- SCORE HINDI ----------
echo ""
echo ">>> [4/4] Scoring Hindi..."
HI_RESULT=$($PYTHON score.py --data_dir "$DATA/hindi" --pred predictions_hindi.csv)
echo "$HI_RESULT"

# ---------- LOG TO RUNLOG.md ----------
echo ""
echo ">>> Appending to RUNLOG.md..."
cat >> RUNLOG.md << EOF

## Run: $TIMESTAMP

**English:**
\`\`\`
$EN_RESULT
\`\`\`

**Hindi:**
\`\`\`
$HI_RESULT
\`\`\`

EOF

echo ""
echo "============================================"
echo "  DONE. Results logged to RUNLOG.md"
echo "============================================"
