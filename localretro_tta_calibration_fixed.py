"""
localretro_tta_calibration_fixed.py
-------------------------------------
Computes calibration CSV for LocalRetro attention TTA.
Pulls errors from calibration_data_localretro.csv (already correct).
UQ = -uncertainty_mean from localretro_attention_tta.csv.
"""

import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

tta = pd.read_csv('localretro_attention_tta.csv')
errors_df = pd.read_csv('calibration_data_localretro.csv')

uq = -tta['uncertainty_mean'].values
error = errors_df['error'].values

out = pd.DataFrame({'uq': uq, 'error': error})

rho, pval = spearmanr(out['uq'], out['error'])
print(f"n molecules : {len(out)}")
print(f"Mean error  : {out['error'].mean():.3f}")
print(f"Mean UQ     : {out['uq'].mean():.3f}")
print(f"Spearman ρ  : {rho:.3f}  (p={pval:.3e})")
print(f"Zeros       : {(out['error']==0).sum()}")

out.to_csv('localretro_tta.csv', index=False)
print("\nSaved: localretro_tta.csv")
