# Experiments Index (MV-HMDA)

## 1. Eedi (real student data)
- Raw cross-table (sparse): `data/eedi_gpt4o_300x1/human_x_model_tau080.csv`
- Proxy labels (densified): `analysis/eedi_proxy_labels.csv`
- Model tags (τ=0.8): 
  - `analysis/eedi_gpt4o_tau08_model_tags.csv`
  - `analysis/eedi_gpt4omini_tau08_model_tags.csv`
- Proxy × model alignment:
  - `analysis/eedi_proxy_x_model_gpt4o_tau08.csv` (≈0.70)
  - `analysis/eedi_proxy_x_model_gpt4omini_tau08.csv` (≈0.80)
- Calibration ablation:
  - `paper_assets/mv-hmda/stage3_calibration_ablation_eedi.csv`
- Baselines (sparse vs proxy):
  - `paper_assets/mv-hmda/stage4_eedi_alignment_baselines.csv`

## 2. RACE (600×5)
- Canonicalized runs: `paper_assets/mv-hmda_race/stage1_canonical_race.csv`
- Proxy labels: `paper_assets/mv-hmda_race/stage2_proxy_labels_race.csv`
- Model-side tags:
  - `paper_assets/mv-hmda_race/stage2_model_tags_race_gpt4omini_tau08.csv`
  - `paper_assets/mv-hmda_race/stage2_model_tags_race_qwen3next80b_tau08.csv`
  - `paper_assets/mv-hmda_race/stage2_model_tags_race_deepseekv3_tau08.csv`
- Alignment summary:
  - `paper_assets/mv-hmda_race/stage3_alignment_summary_race.csv` (all ≈0.86)

## 3. Synthetic (200)
- Generated questions: `synthetic/gen_questions_200.jsonl`
- Simulated LLM runs: `runs/synthetic_llm_runs_200.csv`
- Divergence items: `analysis/synthetic_200/synthetic_separators.json` (137/200)
- Divergence by topic: `analysis/synthetic_200/divergence_by_topic.csv`

## 4. Figures (paper-ready)
- Eedi crosstabs: `paper_assets/eedi_tau08/*.png`
- RACE stage3: `paper_assets/race/stage3/*.png`
- Divergence plots: `figs/eedi_tau08_divergence_*`, `analysis/eedi_tau08_case_report_table.md`


## 5. EdNet-KT1 (sampled)
- Flat logs: `ednet/ednet_kt1_flat_small.csv`
- Proxy labels: `ednet/ednet_proxy_labels.csv`
- Notes: `ednet/RESULTS_EDNET.md`
