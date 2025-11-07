# Appendix: Additional Results & Methods (Auto-Generated)

## EDNet-KT1 Summary
- Overlap items: 36
- Spearman ρ(b vs m): 0.930
- Dev consistency ρ: 0.892  (n=36)
- Test consistency ρ: 0.839 (n=30)
- Gap(dev/test) mean: -0.023 / -0.024

## Cross-Dataset per-question (RACE→EEDI) with Bootstrap CI & Permutation p

| Pair | n | ρ_err [CI95] | |gap| [CI95] | flip [CI95] | p_perm |
|---|---:|---:|---:|---:|---:|
| drobertaB↔eedi_gpt4omini_tau08_joint | 193 | 0.019 [-0.114,0.154] | 0.404 [0.342,0.477] | 0.404 [0.342,0.477] | 0.795 |
| robertaL↔eedi_gpt4o_tau08_joint | 193 | 0.083 [-0.066,0.231] | 0.290 [0.228,0.358] | 0.290 [0.228,0.358] | 0.237 |
| stage3↔stage2_model_tags_eedi_gpt4o_tau08 | 193 | 0.000 [-0.135,0.149] | 0.368 [0.301,0.435] | 0.368 [0.301,0.435] | 0.990 |

### Notes
- ρ: Spearman rank correlation on per-question error rates.
- |gap|: mean absolute difference of per-question error between datasets.
- flip: sign flip rate of centered errors (A vs B).
- CI via bootstrap (B=1000); permutation test (B=2000) shuffling EEDI labels.
