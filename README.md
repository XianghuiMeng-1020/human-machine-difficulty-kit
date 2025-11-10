# 🧠 Human–Machine Difficulty Kit

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A52.0-013243)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-%E2%89%A52.0-150458)](https://pandas.pydata.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.7%2B-F7931E)](https://scikit-learn.org/)
[![StatsModels](https://img.shields.io/badge/StatsModels-0.14%2B-005F87)](https://www.statsmodels.org/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

> A reproducible toolkit for analyzing and aligning **human** vs **machine** difficulty across education datasets (EEDI, RACE, EDNet/KT1, and Synthetic-200).

---

## ✨ Features

- **Unified pipelines** for EEDI, RACE, EDNet/KT1 and Synthetic-200 with consistent CSV outputs  
- **Alignment metrics**: Spearman ρ, error/confidence gaps, calibration, risk–coverage  
- **Statistical modeling**: GLM/GEE, mixed effects, tag-wise analysis, temperature scaling  
- **Model-agnostic scoring API**: swap in any model via `code/src/scoring/interface.py`  
- **Paper-ready tables** under `analysis/` and `paper_assets/` (figures optional)

---

## 🚀 Quick Start

### Prerequisites
- Python **3.10+** (tested up to 3.13)
- macOS/Linux shell environment
- Recommended: create and activate a virtualenv

### Installation

```bash
git clone https://github.com/XianghuiMeng-1020/human-machine-difficulty-kit.git
cd human-machine-difficulty-kit

python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -e .
# optional extras used by dataset scripts / plotting
pip install -r code/requirements.txt
```

### Minimal Repro (tables only)

```bash
# Seed a global alignment table from Synthetic-200 (no external API keys needed)
python code/experiments_20251029/baselines/register_synthetic_divergence.py

# Aggregate global + paper-ready tables
python code/experiments_20251029/make_global_alignment_table.py
python code/experiments_20251029/make_paper_global_table.py
```

**Expected outputs**

```
analysis/global/global_alignment_table.csv
paper_assets/RESULTS_GLOBAL_BASELINES.csv
```

For EEDI / RACE / EDNet pipelines, see **Development → Typical Flow** to populate additional tables under `analysis/`.

---

## 🧩 Core Components

- **Alignment & Metrics** — Spearman ρ (item-level), error/confidence gaps, per-tag aggregates  
- **Calibration & Selective Prediction** — Temperature scaling, reliability summaries, risk–coverage curves  
- **Model-Agnostic Scoring** — Standard interface to plug in any model backend via `scoring.interface.Scorer`

---

## 🔧 Technical Details

- **Metrics**: Spearman ρ, error/confidence gap, calibration AUC/CI, tag-level GLM/GEE  
- **Stats**: generalized linear models, mixed effects, robust SE  
- **Stack**: NumPy, pandas, SciPy, StatsModels, scikit-learn (Matplotlib/Seaborn optional)  
- **Design**: CSV-first; plots are optional and can be regenerated from tables

---

## 📁 Project Structure

```
human-machine-difficulty-kit/
├── .github/
├── code/
        │   ├── EDNET_ANSWER_KEY_STATUS.md
        │   ├── FIGLIST.md
        │   ├── LEAKAGE_AUDIT.md
        │   ├── OVERLAP_CASE_STUDIES.md
    ├── experiments_20251029/
        │       │   │   ├── cl4kt_diff/
        │       │   │   ├── register_all_baselines.py
        │       │   │   ├── register_synthetic_divergence.py
        │   ├── ednet_majority/
        │       │   │   ├── 01_pick_majority_answer.py
        │       │   │   ├── 02_apply_pseudo_contents.py
        │   ├── paper_assets/
        │       │   │   ├── eedi_tau08/
        │       │   │   ├── mv-hmda/
        │       │   │   ├── race/
        │       │   │   ├── synthetic_real/
        │       │   │   ├── BASELINES_REPRO_NOTES.md
        │       │   │   ├── EXPERIMENTS_INDEX.md
        │       │   │   ├── README.md
        │       │   │   ├── SHA256SUMS.txt
        │   ├── scripts/
        │       │   │   ├── 00_build_tidy_from_raw.py
        │       │   │   ├── 01_continuous_alignment_and_logit.py
        │   ├── synthetic/
        │       │   │   ├── gen_questions_200.jsonl
        │       │   │   ├── make_200_questions.py
        │       │   │   ├── summarize_divergence_by_topic.py
        │   ├── SHA256SUMS.txt
        │   ├── align_index.py
        │   ├── align_index_labeled.py
        │   ├── analyze_divergence_full.py
        │   ├── analyze_synthetic_divergence.py
        │   ├── apply_alignment_head.py
        │   ├── apply_alignment_head_pair.py
        │   ├── apply_alignment_head_race_from_eedi.py
        │   ├── apply_joint_head_to_all.py
        │   ├── apply_race_head_to_eedi.py
        │   ├── compare_models.py
        │   ├── csv_to_markdown.py
        │   ├── ednet_compare_two_samples.py
        │   ├── ednet_flatten_any.py
        │   ├── ednet_flatten_from_dir_uid.py
        │   ├── ednet_flatten_from_dir_uid_fixed.py
        │   ├── ednet_flatten_kt1_csv.py
        │   ├── ednet_flatten_kt1_csv_small.py
        │   ├── ednet_label_covaware.py
        │   ├── ednet_make_balanced_subset.py
        │   ├── ednet_make_proxy_labels.py
        │   ├── ednet_make_proxy_labels_big.py
        │   ├── ednet_make_proxy_labels_covaware.py
        │   ├── ednet_make_scaling_table.py
        │   ├── ednet_plot_scaling.py
        │   ├── ednet_scale_run.sh
        │   ├── ednet_summarize_full_proxy.py
        │   ├── eedi_alignment_baselines.py
        │   ├── eedi_batch_analyze.py
        │   ├── eedi_behavior_descriptive.py
        │   ├── eedi_behavior_regression.py
        │   ├── eedi_calibration_ablation.py
        │   ├── eedi_extract_text_features.py
        │   ├── eedi_inspect_crosstab.py
        │   ├── eedi_make_proxy_labels.py
        │   ├── eedi_merge_proxy_two_models.py
        │   ├── eedi_proxy_vs_model.py
        │   ├── eedi_true_alignment_autonorm.py
        │   ├── eedi_true_alignment_from_csv.py
        │   ├── eedi_true_alignment_mapped.py
        │   ├── ... (+42 more files)
    ├── release_20251028/
    ├── reports/
    ├── scripts/
        │   ├── 00_build_from_filelist.py
        │   ├── 00_build_from_filelist.py.bak
        │   ├── 01_continuous_alignment_and_logit.py
        │   ├── 01_continuous_alignment_and_logit.py.bak
        │   ├── 02_misalignment_and_tau.py
        │   ├── 02b_misalignment_significance.py
        │   ├── 03_build_eedi_per_question_from_processed.py
        │   ├── 03_calibration_auc_and_ci.py
        │   ├── 04_by_cogtag_glm.py
        │   ├── 04_race_stage3_reports.py
        │   ├── 05_temp_scaling.py
        │   ├── 05b_finalize_model_summary.py
        │   ├── 06_partial_alignment_control.py
        │   ├── 07_eedi_end_to_end.py
        │   ├── 08_gee_mixed_effects.py
        │   ├── 09_collect_artifacts.py
        │   ├── 10_generalization_gap.py
        │   ├── 10_generalization_gap.py.bak
        │   ├── 11_eedi_mc_infer_hf.py
        │   ├── race_attach_diff.py
        │   ├── report_race.py
        │   ├── report_race.py.bak
        │   ├── sweep_race.sh
    ├── src/
        │       │   │   ├── interface.py
        │       │   │   ├── openai_client_stub.py
        │       │   │   ├── prompts.py
        │   ├── utils/
    ├── Makefile
    ├── filelist.txt
    ├── higher
    ├── pred_candidates.txt
    ├── requirements.txt
    ├── run_4o_grid.sh
    ├── run_all.sh
    ├── runner.py
├── docs/
    ├── EDNET_ANSWER_KEY_STATUS.md
    ├── FIGLIST.md
    ├── LEAKAGE_AUDIT.md
    ├── OVERLAP_CASE_STUDIES.md
├── src/
├── tests/
└── README.md
```

---

## 🎯 Use Cases

- **Educational Research** — quantify human–machine difficulty alignment  
- **Benchmark Diagnostics** — reveal divergence & cognitive gaps beyond accuracy  
- **Model Evaluation** — compare systems via alignment and risk–coverage trade-offs  
- **Ablation Studies** — assess calibration, scaling, and tag-level effects

---

## 🛠️ Development

### Data Layout (example)

```
data/
├── eedi/      # per-item/per-student CSV/JSONL
├── race/      # RACE JSON/CSV with options & keys
├── ednet/     # KT1 flattened logs/aggregates
└── synthetic/ # Synthetic-200 items/splits
```

### Typical Flow

```bash
# 1) Build dataset-level alignment tables
python code/experiments_20251029/eedi_true_alignment_from_csv.py
python code/experiments_20251029/race_alignment_from_proxy.py
python code/experiments_20251029/update_global_with_ednet_full.py

# 2) Aggregate to global + paper tables
python code/experiments_20251029/make_global_alignment_table.py
python code/experiments_20251029/make_paper_global_table.py
```

### Scoring Interface

- Implement `class YourScorer(Scorer)` in `code/src/scoring/` and point dataset scripts to use it.  
- `dummy_client.py` and an OpenAI-style `openai_client_stub.py` are included as examples.

### Testing

```bash
python -m pytest -q
python tests/test_imports.py
```

---

## 📄 License

Released under the MIT License. See [LICENSE](LICENSE).

---

## 📝 Citation

`CITATION.cff` is included for GitHub-native citation. Example BibTeX:

```bibtex
@software{HMDKit_v0_1_0,
  title   = {Human–Machine Difficulty Kit (HMDKit)},
  author  = {Xianghui Meng},
  year    = {2025},
  version = {v0.1.0},
  url     = {https://github.com/XianghuiMeng-1020/human-machine-difficulty-kit},
  note    = {Open-source toolkit for human–machine difficulty alignment}
}
```

---

## 📞 Contact

- **Maintainer**: Xianghui Meng — <xmeng19@illinois.edu>  
- **Issues**: use GitHub Issues in this repository

---

## ⭐ Acknowledgement

If this toolkit helps your research, please consider starring the repo!
