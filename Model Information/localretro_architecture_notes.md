# LocalRetro — Architecture & UQ Technical Notes

## Model architecture

LocalRetro is a template-based single-step retrosynthesis model built on a Message Passing Neural Network (MPNN). Given a product molecule as input, it predicts which reaction template to apply and where to apply it.

The model has two prediction heads:

- **Atom head**: scores each atom in the product as a candidate reaction site
- **Bond head**: scores each bond in the product as a candidate reaction site

Each head outputs a softmax distribution over template classes (124 atom templates + 1 background for the atom head). The template with the highest score at the highest-scoring site is the top-1 prediction.

The model also produces attention scores from its MPNN message-passing layers. These are multi-head attention weights of shape `[batch, heads, max_atoms, max_atoms]` — each head attends over atom pairs during message passing.

---

## Inference pipeline (USPTO-50K test set)

### Step 1 — Raw predictions (`Test.py`)
```
PYTHONPATH=/workspace/LocalRetro/scripts:/workspace/LocalRetro \
python3 Test.py -g cuda:0 -d USPTO_50K
```
Reads `raw_test.csv` (1,001 molecules in `reactants>>product` format). Runs the MPNN forward pass on each product molecule. Outputs raw template predictions (template type, reaction site atom/bond, template ID, confidence score) to `LocalRetro_USPTO_50K_baseline.txt`.

### Step 2 — Decode predictions (`Decode_predictions.py`)
```
PYTHONPATH=/workspace/LocalRetro/scripts:/workspace/LocalRetro \
python3 Decode_predictions.py -d USPTO_50K -k 10
```
Applies each predicted template to the product SMILES to generate actual reactant SMILES. Outputs `LocalRetro_USPTO_50KAfterDecode.txt`. Each line has format:
```
mol_idx \t ('pred_smiles', score) \t ('pred_smiles', score) \t ...
```
Up to 10 predictions per molecule (`-k 10`).

---

## Confidence scores — what they are and what they are not

### What the score is
The confidence score for each prediction is the softmax output of LocalRetro's template classification head at the predicted reaction site. It reflects how strongly the model prefers its top-ranked template over all other templates at that site.

### Verified score distribution (mol 0)
```
top-10 scores: [0.715, 0.058, 0.033, 0.018, 0.016, 0.011, 0.005, 0.004, 0.004, 0.003]
sum of top-10: 0.867
```
The scores do **not** sum to 1.0. The remaining ~0.133 probability mass belongs to templates ranked 11 and beyond, which are not returned by `-k 10`. This means:

- The scores are **not** a complete probability distribution over all templates
- `top_score` is **not** equivalent to `P(top-1 prediction is correct)`
- `top_score` is a **relative confidence** — how much probability mass the model concentrates on its best template compared to the top-10 alternatives

### Correct description for the paper
> "Baseline uncertainty is defined as 1 minus the template confidence score of the top-ranked prediction, where confidence reflects the model's softmax preference for its highest-scoring template. Scores represent partial probability mass over the top-10 decoded templates (mean sum = 0.867 ± σ across the test set) and should not be interpreted as calibrated probabilities."

---

## Baseline UQ method

**UQ signal:** `uq = 1 - top_score`

- Higher `uq` = model less concentrated on its top template = more uncertain
- Lower `uq` = model strongly prefers one template = more confident
- Range: [0.002, 0.966], mean = 0.437

**Error metric:** Tanimoto distance between top-1 predicted reactant and true reactant

- Morgan fingerprints, radius=2, 2048 bits
- True reactants: atom maps stripped, SMILES canonicalized
- Range: [0.0, 1.0] where 0 = exact match, 1 = completely dissimilar
- Distance = 1 − Tanimoto similarity

---

## Verified results

| Metric | Value |
|---|---|
| n molecules | 1,001 |
| Exact matches (error = 0) | 603 (60.2%) |
| Spearman ρ (full dataset) | 0.405 (p = 7.56e-41) |
| Spearman ρ (non-zero errors only, n=398) | −0.011 (p = 0.83) |
| Mean error (full) | 0.162 |
| Mean error (non-zero only) | 0.408 |
| Mean UQ | 0.437 |

### Interpretation
The ρ = 0.405 signal is driven by the model's coarse ability to distinguish easy molecules (low uq, error = 0) from hard molecules (high uq, error > 0). Within the 398 molecules the model got wrong, confidence has essentially no ability to rank prediction difficulty (ρ = −0.011). This motivates TTA — does attention entropy TTA provide genuine fine-grained discrimination within failures?

---

## Alignment verification

Confirmed that `raw_test.csv` row index matches `mol_idx` in `LocalRetro_USPTO_50KAfterDecode.txt`. Format of `raw_test.csv` is `reactants>>product` (single `>>`, no reagents field). Product for mol 0: `CC(C)(C)OC(=O)N1CCC2(CCCc3ccccc32)CC1` — consistent with LocalRetro's decode output for the same index.

---

## Known limitations

1. **Score incompleteness:** Top-10 scores sum to ~0.867, not 1.0. The full softmax distribution is not recovered.
2. **Top-1 only:** Error is computed against the single best prediction. LocalRetro may have the correct answer in top-2 through top-10.
3. **Zero inflation:** 60.2% exact match rate compresses error distribution dynamic range. Calibration metrics (miscalibration area, NLL) are influenced by the large mass at error = 0.
4. **Multi-component SMILES:** Morgan fingerprints of multi-fragment SMILES (e.g. `A.B`) are computed on the combined graph, which may not fully reflect structural similarity of individual fragments.
