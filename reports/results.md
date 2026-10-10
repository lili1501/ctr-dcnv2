# Results (SYNTHETIC DATA - validates the code, not a research claim)

Held-out test split, 30,037 rows. Replace with numbers from a real Criteo run before citing.

| Model | AUC | Log loss | ECE |
|---|---|---|---|
| Constant base rate | - | 0.5629 | - |
| LightGBM | 0.7768 | 0.4636 | 0.0108 |
| Deep-only (no cross layers), 3 seed(s) | 0.7764 | 0.4638 | 0.0107 |
| DCN-V2, 3 seed(s) | 0.7812 | 0.4597 | 0.0098 |
| Oracle (true probabilities) | 0.8040 | 0.4399 | - |

* DCN-V2 vs LightGBM: log loss +0.84% better, AUC +0.0044.
* Paired bootstrap (DCN-V2 seed 0 minus LightGBM), 95% CI: AUC diff +0.0028 to +0.0064; log-loss diff -0.0056 to -0.0027.
* ECE = expected calibration error over 15 equal-count bins (lower is better).
