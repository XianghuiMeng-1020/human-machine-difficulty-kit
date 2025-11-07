# Stage 4: Divergence Attribution (Eedi)

- source: `stage1_canonical.csv`
- human side: `stage2_proxy_labels_eedi.csv`
- model side: `stage2_model_tags_eedi_*.csv`
- output: `stage4_divergence_attr_eedi.csv`

Columns:
- `feature`: behavior-level feature
- `mean_div0`: feature mean when human–model difficulty aligned
- `mean_div1`: feature mean when human–model difficulty diverged
- `diff`: mean_div1 - mean_div0

Key findings (from this run):
- `mini_says_hard` ≈ 0.99 when diverged → weak-model-induced difficulty is the main trigger.
- `any_high_conf` = 1.0 when diverged → misalignment is caused by overconfident models, not by mutual uncertainty.
