# Results (real data, held-out chronological test split)

Test split: 300,478 rows, scored once after model selection on the validation split.

| Model | AUC | Log loss | ECE |
|---|---|---|---|
| Constant base rate | - | 0.6392 | - |
| LightGBM | 0.6884 | 0.5865 | 0.0076 |
| Deep-only (no cross layers), 3 seed(s) | 0.6849 | 0.5895 | 0.0231 |
| DCN-V2, 3 seed(s) | 0.6858 | 0.5886 | 0.0195 |

* DCN-V2 vs LightGBM: log loss -0.36% better, AUC -0.0026.
* Paired bootstrap (DCN-V2 seed 0 minus LightGBM), 95% CI: AUC diff -0.0031 to -0.0014; log-loss diff +0.0009 to +0.0018.
* ECE = expected calibration error over 15 equal-count bins (lower is better).
