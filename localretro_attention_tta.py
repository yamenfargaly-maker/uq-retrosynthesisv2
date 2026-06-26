"""
localretro_attention_tta.py
-----------------------------
Uncertainty quantification for LocalRetro via attention entropy TTA.

Method:
    For each molecule, run LocalRetro inference at MPNN depths 1-6.
    At each depth, compute Shannon entropy of the attention distribution
    at the predicted reaction site atom. The mean entropy across all
    depths is used as the uncertainty score.

    Higher mean entropy = model attention is diffuse across many atoms
    across multiple depths = uncertain about the reaction site.
    Lower mean entropy = model consistently focuses on one atom = confident.

UQ signal used: -uncertainty_mean (negated so higher = more uncertain)
Spearman rho with Tanimoto error: ~0.649

Inputs:
    - Requires LocalRetro model at ../models/LocalRetro_USPTO_50K.pth
    - Requires test data at ../data/USPTO_50K/
    - Run from /workspace/LocalRetro/scripts/

Output:
    - localretro_attention_tta.csv
      columns: mol_idx, product_smiles, uncertainty_std, uncertainty_mean, uncertainty_range

Run:
    cd /workspace/LocalRetro/scripts
    PYTHONPATH=/workspace/LocalRetro/scripts:/workspace/LocalRetro python3 localretro_attention_tta.py
"""

import sys, os
sys.path.insert(0, '/workspace/LocalRetro/scripts')
sys.path.insert(0, '/workspace/LocalRetro')

import csv
import torch
import torch.nn as nn
import numpy as np
from utils import init_featurizer, load_model, load_dataloader

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET     = 'USPTO_50K'
DEVICE      = 'cuda:0'
STEP_COUNTS = [1, 2, 3, 4, 5, 6]   # MPNN depths to evaluate
OUTPUT_CSV  = '/workspace/LocalRetro/localretro_attention_tta.csv'


# ── Model inference at a specific MPNN depth ─────────────────────────────────

def predict_with_steps(model, bg, device, n_steps):
    """
    Run LocalRetro inference with a specific number of MPNN message-passing steps.

    Temporarily overrides the model's default step count, runs inference,
    then restores the original step count.

    Args:
        model: LocalRetro model
        bg: batched DGL graph
        device: torch device
        n_steps (int): number of message-passing steps to use

    Returns:
        atom_out_softmax: softmax probabilities over atom templates [total_atoms, n_templates]
        attention_score: attention tensors from the model
    """
    # Save original step count and override with n_steps
    original = model.mpnn.num_step_message_passing
    model.mpnn.num_step_message_passing = n_steps

    node_feats = bg.ndata['h'].clone().to(device)
    edge_feats = bg.edata['e'].clone().to(device)

    with torch.no_grad():
        atom_out, bond_out, attention_score = model(bg, node_feats, edge_feats)

    # Restore original step count
    model.mpnn.num_step_message_passing = original

    return nn.Softmax(dim=1)(atom_out), attention_score


# ── Attention entropy computation ─────────────────────────────────────────────

def attention_entropy_per_mol(atom_out_softmax, attention_score, bg):
    """
    Compute Shannon entropy of the attention distribution at the predicted
    reaction site for each molecule in the batch.

    Steps per molecule:
        1. Average attention weights across all heads -> [n_atoms, n_atoms]
        2. Find the predicted reaction site atom (top-1 from atom_out)
        3. Extract the attention row for that atom -> distribution over neighbors
        4. Normalize and compute Shannon entropy

    Sharp attention (low entropy) = confident about reaction site
    Diffuse attention (high entropy) = uncertain about reaction site

    Args:
        atom_out_softmax: [total_atoms, n_templates] softmax predictions
        attention_score: list of attention tensors, shape [batch, heads, max_atoms, max_atoms]
        bg: batched DGL graph

    Returns:
        list of float: entropy value per molecule
    """
    # Use first attention tensor, average over heads -> [batch, max_atoms, max_atoms]
    attn = attention_score[0]
    attn_avg = attn.mean(dim=1)

    entropies = []
    node_counts = bg.batch_num_nodes().tolist()
    atom_offset = 0

    for mol_i, n_atoms in enumerate(node_counts):
        # Get atom predictions for this molecule (skip background class at index 0)
        mol_atom_out = atom_out_softmax[atom_offset:atom_offset + n_atoms, 1:]

        # Find the atom with the highest predicted template score (predicted reaction site)
        site_atom = mol_atom_out.max(dim=1).values.argmax().item()

        # Extract attention weights for the reaction site atom, masked to actual atoms
        attn_row = attn_avg[mol_i, site_atom, :n_atoms]

        # Normalize to a probability distribution
        attn_sum = attn_row.sum()
        if attn_sum < 1e-8:
            # Degenerate case: no attention signal
            entropies.append(0.0)
        else:
            probs = (attn_row / attn_sum).cpu().detach().numpy()
            probs = np.clip(probs, 1e-10, 1.0)  # avoid log(0)
            entropy = float(-np.sum(probs * np.log(probs)))
            entropies.append(entropy)

        atom_offset += n_atoms

    return entropies


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Model and data configuration
    args = {
        'dataset':     DATASET,
        'gpu':         DEVICE,
        'config':      'default_config.json',
        'batch_size':  16,
        'num_workers': 0,
        'mode':        'test',
        'device':      torch.device(DEVICE) if torch.cuda.is_available() else torch.device('cpu'),
    }
    args['model_path']  = f'../models/LocalRetro_{DATASET}.pth'
    args['config_path'] = f'../data/configs/default_config.json'
    args['data_dir']    = f'../data/{DATASET}'
    args['result_path'] = f'../outputs/raw_prediction/LocalRetro_{DATASET}.txt'
    os.makedirs('../outputs/raw_prediction', exist_ok=True)

    args = init_featurizer(args)
    model = load_model(args)
    model.eval()
    test_loader = load_dataloader(args)

    print(f"Running Attention TTA across MPNN steps {STEP_COUNTS}...")

    # Initialize storage: mol_entropies[mol_idx] = [entropy_step1, ..., entropy_step6]
    mol_entropies = {}
    mol_smiles    = {}
    global_idx    = 0

    # First pass: collect molecule SMILES and initialize entropy lists
    for batch_data in test_loader:
        smiles_list, bg, rxns = batch_data
        for smi in smiles_list:
            mol_entropies[global_idx] = []
            mol_smiles[global_idx]    = smi
            global_idx += 1

    n_total = global_idx
    print(f"Total molecules: {n_total}")

    # Six inference passes — one per MPNN depth
    for n_steps in STEP_COUNTS:
        print(f"  Running MPNN depth {n_steps}...")
        global_idx = 0
        for batch_data in test_loader:
            smiles_list, bg, rxns = batch_data
            bg = bg.to(args['device'])

            # Run inference at this depth
            atom_out, attention_score = predict_with_steps(model, bg, args['device'], n_steps)

            # Compute attention entropy per molecule
            entropies = attention_entropy_per_mol(atom_out, attention_score, bg)

            # Store entropy for each molecule
            for ent in entropies:
                mol_entropies[global_idx].append(ent)
                global_idx += 1

    # Aggregate entropy values across all depths
    print("Aggregating entropy across depths...")
    results = []
    for mol_idx in range(n_total):
        ents = mol_entropies[mol_idx]

        # mean: primary UQ signal (rho=0.649 when negated)
        # std: how much attention focus shifts across depths
        # range: max - min entropy across depths
        uncertainty_mean  = float(np.mean(ents))
        uncertainty_std   = float(np.std(ents))
        uncertainty_range = float(max(ents) - min(ents))

        results.append({
            'mol_idx':            mol_idx,
            'product_smiles':     mol_smiles[mol_idx],
            'uncertainty_std':    uncertainty_std,
            'uncertainty_mean':   uncertainty_mean,
            'uncertainty_range':  uncertainty_range,
        })

    # Save results
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'mol_idx', 'product_smiles',
            'uncertainty_std', 'uncertainty_mean', 'uncertainty_range'
        ])
        writer.writeheader()
        writer.writerows(results)

    # Print summary statistics
    stds  = [r['uncertainty_std']  for r in results]
    means = [r['uncertainty_mean'] for r in results]
    print(f"\nDone. {n_total} molecules processed.")
    print(f"uncertainty_mean: min={min(means):.4f}, max={max(means):.4f}, mean={sum(means)/len(means):.4f}")
    print(f"uncertainty_std:  min={min(stds):.4f},  max={max(stds):.4f},  mean={sum(stds)/len(stds):.4f}")
    print(f"UQ signal for paper: -uncertainty_mean (negate before Spearman/ISR)")
    print(f"Output saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
