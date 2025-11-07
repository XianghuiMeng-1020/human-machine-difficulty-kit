set -euo pipefail

# fixed env
export USE_IMAGE=0
export TOP_LOGK=20
export NORM_SCOPE=letters
export STRICT_LETTER=1

# A: first_raw + LETTER_TOKENS=3 + T=0.0
export CONF_MODE=first_raw
export LETTER_TOKENS=3
export GEN_T=0.0
python runner.py score --dataset eedi --split test \
  --model openai:gpt-4o --rounds 1 \
  --out outputs/eedi_gpt4o_300x1_FR_L3_T00

# B: first_letter + LETTER_TOKENS=1 + T=0.0
export CONF_MODE=first_letter
export LETTER_TOKENS=1
export GEN_T=0.0
python runner.py score --dataset eedi --split test \
  --model openai:gpt-4o --rounds 1 \
  --out outputs/eedi_gpt4o_300x1_FL_L1_T00

# C: first_raw + LETTER_TOKENS=1 + T=0.0
export CONF_MODE=first_raw
export LETTER_TOKENS=1
export GEN_T=0.0
python runner.py score --dataset eedi --split test \
  --model openai:gpt-4o --rounds 1 \
  --out outputs/eedi_gpt4o_300x1_FR_L1_T00

# D: first_letter + LETTER_TOKENS=3 + T=0.05
export CONF_MODE=first_letter
export LETTER_TOKENS=3
export GEN_T=0.05
python runner.py score --dataset eedi --split test \
  --model openai:gpt-4o --rounds 1 \
  --out outputs/eedi_gpt4o_300x1_FL_L3_T005
