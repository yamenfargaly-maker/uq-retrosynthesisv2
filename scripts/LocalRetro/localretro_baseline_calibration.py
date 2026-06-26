"""
localretro_baseline_calibration.py
------------------------------------
Computes uncertainty quantification (UQ) scores and Tanimoto errors
for the LocalRetro baseline model on the USPTO-50K validation set.

Inputs:
    - LocalRetro_USPTO_50KAfterDecode.txt : decoded predictions from Decode_predictions.py
                                            format: mol_idx \t (pred_smiles, score) \t ...
    - raw_test.csv                        : USPTO-50K test set in reaction SMILES format
                                            column: 'reactants>reagents>production'

Output:
    - localretro_baseline.csv : per-molecule UQ score and Tanimoto error
                                columns: 'uq', 'error'

UQ method:
    Baseline uncertainty = 1 - top_score
    where top_score is the softmax confidence of the highest-ranked template prediction.
    Higher uncertainty = lower model confidence in its top prediction.

Error metric:
    Tanimoto distance between predicted reactant (top-1) and true reactant.
    Computed using Morgan fingerprints (radius=2, 2048 bits).
    Distance = 1 - Tanimoto similarity. Range: [0, 1].
    0 = identical molecules, 1 = completely dissimilar.

    True reactants are extracted from raw_test.csv with atom maps stripped
    and SMILES canonicalized before comparison to ensure fair evaluation.
"""

import ast
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


# ── Helper functions ─────────────────────────────────────────────────────────

def clean_smiles(smi):
    """
    Strip atom map numbers and canonicalize a SMILES string.

    Atom maps (e.g. [CH3:1]) are used in reaction SMILES to track atoms
    but should be removed before molecular comparison to avoid Tanimoto
    artifacts from RDKit parsing atom-mapped vs canonical SMILES differently.

    Args:
        smi (str): Input SMILES (may be atom-mapped)

    Returns:
        str: Canonical SMILES with atom maps removed, or original if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        # Remove atom map numbers from all atoms
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smi


def tanimoto_dist(s1, s2):
    """
    Compute Tanimoto distance between two SMILES strings.

    Uses Morgan fingerprints (radius=2, 2048 bits) as molecular representation.
    Returns 1.0 (maximum distance) if either SMILES is invalid.

    Args:
        s1 (str): SMILES string for molecule 1
        s2 (str): SMILES string for molecule 2

    Returns:
        float: Tanimoto distance in [0, 1]
    """
    try:
        m1 = Chem.MolFromSmiles(s1)
        m2 = Chem.MolFromSmiles(s2)
        # Return max distance if either molecule fails to parse
        if m1 is None or m2 is None:
            return 1.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius=2, nBits=2048)
        # Convert similarity to distance
        return 1 - DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 1.0


# ── Load true reactants from raw_test.csv ───────────────────────────────────

# raw_test.csv contains reactions in 'reactants>reagents>production' format
# We extract the reactant side (left of '>>') as the ground truth
raw = pd.read_csv('raw_test.csv')
raw.columns = ['rxn']

# Extract true reactants, strip atom maps, and canonicalize SMILES
# This ensures fair Tanimoto comparison against model predictions
true_reactants = raw['rxn'].apply(
    lambda x: clean_smiles(x.split('>>')[0])
).tolist()


# ── Parse decoded predictions from LocalRetro ───────────────────────────────

# Each line in the decoded predictions file has format:
#   mol_idx \t ('pred_smiles', score) \t ('pred_smiles', score) \t ...
# We only use the top-1 prediction (first entry after mol_idx)

results = []
with open('LocalRetro_USPTO_50KAfterDecode.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        mol_idx = int(parts[0])

        # Extract top-1 prediction tuple (smiles, score)
        top_smiles = None
        top_score = None
        if len(parts) > 1:
            try:
                # Each prediction is stored as a string representation of a tuple
                tup = ast.literal_eval(parts[1])
                top_smiles = tup[0]  # predicted reactant SMILES
                top_score = tup[1]   # model confidence score (softmax output)
            except Exception:
                pass  # Leave as None if parsing fails

        results.append({
            'mol_idx': mol_idx,
            'top_pred': top_smiles,
            'score': top_score
        })

# Sort by mol_idx to ensure alignment with true_reactants
df = pd.DataFrame(results).sort_values('mol_idx').reset_index(drop=True)

# Add true reactants (row order matches mol_idx order)
df['true_reactant'] = true_reactants


# ── Compute Tanimoto errors ──────────────────────────────────────────────────

# For each molecule, compare top-1 prediction to true reactant
# If no valid prediction exists, assign maximum error (1.0)
errors = []
for _, row in df.iterrows():
    if row['top_pred'] is None:
        errors.append(1.0)
    else:
        errors.append(tanimoto_dist(row['top_pred'], row['true_reactant']))

df['error'] = errors


# ── Compute UQ scores ────────────────────────────────────────────────────────

# Baseline UQ: uncertainty = 1 - confidence
# The model's top template score is its confidence in the prediction.
# We invert it so that high uncertainty = low confidence = high expected error.
df['uq'] = 1 - df['score']


# ── Evaluate calibration ─────────────────────────────────────────────────────

rho, pval = spearmanr(df['uq'], df['error'])
print(f"n molecules : {len(df)}")
print(f"Mean error  : {df['error'].mean():.3f}")
print(f"Mean UQ     : {df['uq'].mean():.3f}")
print(f"Spearman ρ  : {rho:.3f}  (p={pval:.3e})")
print("Ideal: ρ > 0 (higher uncertainty should correlate with higher error)")


# ── Save output ──────────────────────────────────────────────────────────────

# Save per-molecule UQ and error for downstream calibration analysis (ISR, plots)
out = df[['uq', 'error']]
out.to_csv('localretro_baseline.csv', index=False)
print("\nSaved: localretro_baseline.csv")
print("Columns: uq (uncertainty score), error (Tanimoto distance)")
