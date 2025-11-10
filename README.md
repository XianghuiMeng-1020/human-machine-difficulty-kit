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
├── code/
│   ├── experiments_20251029/
│   │   ├── make_global_alignment_table.py
│   │   ├── make_paper_global_table.py
│   │   ├── eedi_*.py / race_*.py / ednet_*.py      # dataset pipelines
│   │   └── plot_*.py                               # optional plotting
│   ├── src/scoring/                                # scoring interface & clients
│   └── requirements.txt                            # script-level extras
├── src/hmdkit/                                     # importable package
├── analysis/                                       # generated CSV tables
├── paper_assets/                                   # paper-consumable tables
├── figs/                                           # optional figures (paper)
├── figures/                                        # optional figures (analysis)
├── tests/
├── CITATION.cff
├── LICENSE
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
