#!/usr/bin/env bash
set -euo pipefail

# Common env
export USE_IMAGE=0 TOP_LOGK=20 STRICT_LETTER=1 LETTER_TOKENS=3 CONF_MODE=first_letter NORM_SCOPE=letters
# Light rounds for the sweep so it finishes fast
ROUNDS_SWEEP="${ROUNDS_SWEEP:-1}"

# Temperatures to try
TEMPS=("0.10" "0.12" "0.15" "0.18" "0.20" "0.25" "0.30")

for T in "${TEMPS[@]}"; do
  export GEN_T="$T"
  tag="${T/./}"              # 0.10 -> 010, etc.
  out="outputs/race_gpt4omini_600x2_T${tag}"

  # Skip if we already have a complete summary for this temp
  if [ -f "$out/overall.csv" ]; then
    echo "[skip] $out already summarized"
    continue
  fi

  # Run the scorer (rounds=1 for sweep)
  python runner.py score --dataset race --split test \
    --model openai:gpt-4o-mini-2024-07-18 --rounds "$ROUNDS_SWEEP" \
    --out "$out"

  # Attach difficulty (robust mapper that infers from question_id)
  python scripts/race_attach_diff.py \
    --scores "$out/scores.csv" \
    --out    "$out/scores_wdiff.csv"

  # Produce the mini report (uses p_correct.mean() as Mean Conf)
  PYTHONPATH=. python scripts/report_race.py \
    --scores "$out/scores_wdiff.csv" \
    --outdir "$out"

  # Gentle cooldown to avoid bursts on the API
  sleep 0.5
done
