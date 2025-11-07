# Baselines Reproduction Notes

We implemented three representative baselines to match our MV-HMDA setting:

1. **LLM-based difficulty (BEA24-style)**
   - Scripts: `baselines/bea24_llm_diff/*.py`
   - Data source: unified item table at `baselines/bea24_llm_diff/out/all_items_raw.csv`
   - Feature variants:
     - LLM-like shallow features: `02_make_llm_features_stub.py`
     - Text-only BoW features: `22_featurize_bow.py` + `23_train_bow_logreg.py`
   - Result is registered by running: `python baselines/register_all_baselines.py`

2. **KT difficulty-aware (CL4KT-style, EdNet-KT1)**
   - Flattened logs: `analysis/ednet_flat_u200.csv` (and u500/u1000 for scaling)
   - Sessions: `baselines/cl4kt_diff/ednet_u200_sessions.csv`
   - Difficulty table: `baselines/cl4kt_diff/ednet_u200_items.csv`
   - We provide both torch-free logistic version and strict step-wise version:
     - `02_train_cl4kt_stub_notorch.py`
     - `03_train_strict_logreg.py`
   - Results are also registered via `baselines/register_all_baselines.py`.

3. **Calibration / temperature-style baselines**
   - Script: `eedi_calibration_ablation.py`
   - Registered into the global table by `update_global_with_eedi_true.py` and finally `baselines/register_all_baselines.py`.

All baseline outputs end up in a single file:
- `analysis/global/global_alignment_table.csv`

This file is then filtered into a paper-ready table:
- `paper_assets/RESULTS_GLOBAL_BASELINES.csv`
