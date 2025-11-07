# RACE / stage3 结果摘要

## Confidence Discrimination (ROC AUC)
- **model**: stage3
- **ROC_AUC**: 0.443

## Calibration
- **model**: stage3
- **ECE(15bins)**: 0.007
- **Brier**: 0.184

## Correlation (point estimates)
- **rho(human_diff, err)**: -0.042
- **tau(human_diff, err)**: -0.042
- **rho(human_diff, 1-meanconf)**: 0.033
- **tau(human_diff, 1-meanconf)**: 0.027
- **n_err**: 600
- **n_1mconf**: 600

## Correlation (bootstrap 95% CI)
- **Spearman(err)**: -0.008 [-0.088, 0.074]
- **Kendall(err)**: -0.008 [-0.085, 0.072]
- **Spearman(1m)**: 0.032 [-0.042, 0.108]
- **Kendall(1m)**: 0.025 [-0.033, 0.084]
- **n(err)**: 600
- **n(1mconf)**: 600

## Risk-Coverage (head)

| model   |    covered |   cum_acc |   mean_conf |
|:--------|-----------:|----------:|------------:|
| stage3  | 0.00166667 |         0 |    0.253369 |
| stage3  | 0.00333333 |         0 |    0.252987 |
| stage3  | 0.005      |         0 |    0.252775 |
| stage3  | 0.00666667 |         0 |    0.25255  |
| stage3  | 0.00833333 |         0 |    0.252401 |
| stage3  | 0.01       |         0 |    0.2523   |
| stage3  | 0.0116667  |         0 |    0.252227 |
| stage3  | 0.0133333  |         0 |    0.252158 |
| stage3  | 0.015      |         0 |    0.252087 |
| stage3  | 0.0166667  |         0 |    0.252029 |
