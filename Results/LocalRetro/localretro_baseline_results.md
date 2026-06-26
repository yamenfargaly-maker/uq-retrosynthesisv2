# LocalRetro Baseline — Calibration Results

## Method
- **Model:** LocalRetro (GNN template-based retrosynthesis)
- **UQ method:** Baseline — template score confidence (1 - top prediction score)
- **Dataset:** USPTO-50K validation set (n=1,001)
- **Error metric:** Tanimoto distance (top-1 prediction vs true reactant)
- **Calibration framework:** Rasmussen et al. 2023

---

## Metrics

| Metric | Value |
|---|---|
| Spearman ρ (observed) | 0.41 |
| Spearman ρ (simulated) | 0.54 ± 0.02 |
| Miscalibration area | 0.29 |
| NLL (observed) | 0.75 |
| NLL (simulated) | 0.39 ± 0.02 |
| R² (RMSE vs RMV) | 0.85 |
| Slope | 0.32 |
| Intercept | 0.13 |
| var(Z) | 1.54 |
| mean(Z) | 0.42 |

---

## Interpretation

**Discrimination (Spearman ρ = 0.41):** The baseline template score has moderate rank correlation with prediction error — the model assigns higher uncertainty to harder molecules more often than not, but the signal is weak.

**Calibration (miscalibration area = 0.29):** The calibration curve sits well below the diagonal, meaning the model is overconfident — it assigns lower uncertainty than the actual error rate warrants across most of the uncertainty range.

**RMSE vs RMV (R² = 0.85, slope = 0.32):** Strong linear relationship between predicted uncertainty and actual error variance, but the slope of 0.32 (well below 1.0) confirms the model underestimates its own uncertainty magnitude.

**Z-score distribution:** The distribution is sharply peaked around 0 with var(Z) = 1.54 and mean(Z) = 0.42, indicating the errors are not well-explained by the uncertainty estimates alone — the model is systematically overconfident.

---

## Output files
- `localretro_baseline.csv` — per-molecule UQ scores and Tanimoto errors
- `localretro_baseline_calibration.py` — script used to generate the CSV
- Calibration curve plot (miscalibration area = 0.29)
- Z-score distribution plot
- RMSE vs RMV plot (R² = 0.85)
