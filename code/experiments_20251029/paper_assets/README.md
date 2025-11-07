# Paper Assets Index

## Eedi (τ=0.8)
- Figures:  
  - `eedi_gpt4o_crosstab_t0.8.png`, `eedi_gpt4omini_crosstab_t0.8.png`  
  - `eedi_tau08_divergence_summary_p.png`, `eedi_tau08_divergence_summary_acc.png`  
  - `eedi_tau08_deltaP_miniHard_top10_dp.png`, `eedi_tau08_deltaP_miniEasy_top10_dp.png`
- Tables & Notes:  
  - `eedi_tau08_divergence_metrics.csv`, `eedi_tau08_divergence_sample_table.csv`  
  - `eedi_tau08_case_report_table.(csv|md)`, `eedi_tau08_case_notes.md`
- Text snippet: `eedi_tau08/RESULTS.md`

## RACE
- Per-model folders under `paper_assets/race/{gpt4omini,qwen3next80b,deepseekv3}`:  
  - `overall.csv`, `by_human.csv`, `datamap_counts.csv`, `datamap.png`, `reliability.png`
- Text snippet: `race/RESULTS_RACE.md`

## One-line takeaways
- Eedi：两类模型分歧呈互补——mini 判难时 GPT-4o 更准但更谨慎；mini 判易时 mini 高自信且显著更准，提示题型捷径/适配效应。  
- RACE：各模型与人类难度方向一致（middle > high），强模型 Easy 区更大、Ambiguous 更小，校准更稳。

## Pack command
在项目根目录执行：
