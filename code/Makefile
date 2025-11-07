PY ?= python
MODELS ?= stage3 robertaL drobertaB

.PHONY: all infer build analytics misalign sig calib ci figs clean

all: infer build analytics sig calib
	@echo "[DONE] 全流程完成。"

infer:
	@echo "[RUN] 推理（按需跳过已有）"
	@DEVICE=$(DEVICE) MODELS="$(MODELS)" $(PY) - <<'PY'
import os, sys
os.system("./run_all.sh >/dev/null")  # 复用 run_all.sh 的推理段（会自动跳过）
PY

build:
	@$(PY) scripts/00_build_from_filelist.py

analytics:
	@$(PY) scripts/01_continuous_alignment_and_logit.py
	@$(PY) scripts/02_misalignment_and_tau.py

sig:
	@$(PY) scripts/02b_misalignment_significance.py

calib:
	@$(PY) scripts/03_calibration_auc_and_ci.py

figs:
	@ls -1 figures | sed 's/^/[FIG] /'

clean:
	@rm -f tables/*.csv tables/*.tex figures/*.png
	@echo "[CLEAN] tables/ 和 figures/ 已清理"
