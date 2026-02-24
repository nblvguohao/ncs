#!/usr/bin/env python3
"""
AlphaFold-Multimer Validation for GPCR-Gα Interface Analysis
Run this script in Google Colab with A100 GPU

Instructions:
1. Upload this script to Colab
2. Set Runtime -> GPU (A100 if available)
3. Run all cells
"""

import os
import sys
import subprocess
import time
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser
import scipy.stats as stats

# Set up environment
print("🔧 Setting up environment...")
!pip install colabfold[alphafold] --quiet
!pip install biopython --quiet
!pip install matplotlib seaborn pandas numpy scipy --quiet

# Check GPU
print("\n🎮 GPU Status:")
!nvidia-smi

# Download data package
print("\n📦 Downloading data package...")
!mkdir -p /content/data /content/results
!wget -q https://github.com/nblvguohao/ncs/archive/main.zip -O ncs_main.zip
!unzip -q ncs_main.zip
!mv ncs-main/* /content/
!rm -rf ncs-main ncs_main.zip

print("✅ Data downloaded and extracted")
print("\n📁 Available files:")
!ls -la /content/data/
!ls -la /content/results/alphafold_validation/fasta_inputs/ | head -5

# Run AlphaFold-Multimer predictions
print("\n🚀 Starting AlphaFold-Multimer predictions...")
print("⏱️  Expected time: 2-8 hours depending on GPU")

fasta_dir = Path("/content/results/alphafold_validation/fasta_inputs")
output_dir = Path("/content/results/alphafold_validation/af_predictions")
output_dir.mkdir(exist_ok=True, parents=True)

cmd = [
    "colabfold_batch",
    "--model-type", "alphafold2_multimer_v3",
    "--num-recycle", "3",
    "--num-models", "5",
    "--amber",
    "--gpu-devices", "0",
    str(fasta_dir),
    str(output_dir)
]

start_time = time.time()
result = subprocess.run(cmd, capture_output=True, text=True)
elapsed = time.time() - start_time

if result.returncode == 0:
    print(f"✅ Predictions completed in {elapsed/3600:.1f} hours")
    print(f"📁 Results saved to: {output_dir}")
else:
    print(f"❌ Error: {result.stderr}")

print("\n📋 Generated files:")
!ls -la {output_dir}/*.pdb | head -10

# Analysis functions
def extract_contacts_from_pdb(pdb_path, receptor_chain="A", galpha_chain="B", cutoff=8.0):
    """Extract receptor-Gα contacts from PDB structure."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    
    receptor_atoms = []
    galpha_atoms = []
    
    for model in structure:
        for chain in model:
            if chain.id == receptor_chain:
                for residue in chain:
                    if residue.id[0] == " ":  # Skip heteroatoms
                        for atom in residue:
                            if atom.element != "H":
                                receptor_atoms.append((residue, atom))
            elif chain.id == galpha_chain:
                for residue in chain:
                    if residue.id[0] == " ":
                        for atom in residue:
                            if atom.element != "H":
                                galpha_atoms.append((residue, atom))
    
    # Find contacts
    contacts = []
    for res_rec, atom_rec in receptor_atoms:
        for res_ga, atom_ga in galpha_atoms:
            distance = atom_rec - atom_ga
            if distance <= cutoff:
                contacts.append((res_rec.id[1], res_ga.id[1], distance))
    
    return contacts

def map_receptor_residues_to_bw(entry_name, contacts, bw_cache):
    """Map receptor residue numbers to BW positions."""
    if entry_name not in bw_cache:
        return {}
    
    residue_data = bw_cache[entry_name]
    bw_mapping = {}
    
    for res in residue_data:
        seq_num = res.get("sequence_number")
        bw_label = res.get("display_generic_number") or res.get("generic_number", "")
        if isinstance(bw_label, dict):
            bw_label = bw_label.get("label", "")
        if seq_num and bw_label and "x" in str(bw_label):
            bw_mapping[int(seq_num)] = str(bw_label)
    
    return bw_mapping

# Load data and analyze
print("\n🔬 Analyzing predicted structures...")

# Add to path
sys.path.append('/content')

from leakageguard.data.dataset import GPCRDataset
from leakageguard.features.bw_site import load_bw_cache, GP_CONTACT_SITES

dataset = GPCRDataset()
dataset.load()
bw_cache = load_bw_cache()

# Load FDR results
fdr_file = Path("/content/results/bw_site_fdr_results.csv")
if fdr_file.exists():
    fdr_results = pd.read_csv(fdr_file)
    fdr_sites = fdr_results[fdr_results["fdr"] < 0.05]["bw_label"].tolist()
    print(f"✅ Loaded {len(fdr_sites)} FDR-significant BW sites")
else:
    print("⚠️  FDR results not found, using all BW sites")
    fdr_sites = list(GP_CONTACT_SITES.keys())

# Analyze structures
target_receptors = [
    "opn4_human",    # Gq
    "hrh1_human",    # Gq
    "acm1_human",    # Gq
    "5ht2a_human",   # Gq
    "oprm_human",    # Gi
    "acm4_human",    # Gi
    "opsd_human",    # Gi
    "adrb2_human",   # Gs
    "glr_human",     # Gs
    "drd1_human",    # Dual
]

results = []
pdb_dir = Path("/content/results/alphafold_validation/af_predictions")

for entry_name in target_receptors:
    pdb_file = pdb_dir / f"{entry_name}_ranked_0.pdb"
    if not pdb_file.exists():
        print(f"⚠️  Missing PDB for {entry_name}")
        continue
    
    print(f"  📊 {entry_name}...")
    
    # Extract contacts
    contacts = extract_contacts_from_pdb(pdb_file)
    
    # Map to BW positions
    bw_mapping = map_receptor_residues_to_bw(entry_name, contacts, bw_cache)
    
    # Count contacts at BW sites
    bw_contacts = {}
    for rec_res, ga_res, dist in contacts:
        if rec_res in bw_mapping:
            bw_pos = bw_mapping[rec_res]
            if bw_pos not in bw_contacts:
                bw_contacts[bw_pos] = 0
            bw_contacts[bw_pos] += 1
    
    # Get coupling info
    coupling = "Unknown"
    for i, entry in enumerate(dataset.entry_names):
        if entry == entry_name:
            if dataset.y[i, 1] > 0:  # Gq
                coupling = "Gq"
            elif dataset.y[i, 0] > 0:  # Gs
                coupling = "Gs"
            elif dataset.y[i, 2] > 0:  # Gi
                coupling = "Gi"
            elif dataset.y[i, 3] > 0:  # G12
                coupling = "G12"
            break
    
    results.append({
        "entry_name": entry_name,
        "coupling": coupling,
        "total_contacts": len(contacts),
        "bw_contacts": len(bw_contacts),
        "bw_contact_sites": list(bw_contacts.keys()),
        "fdr_bw_contacts": len([bw for bw in bw_contacts.keys() if bw in fdr_sites]),
        "non_fdr_bw_contacts": len([bw for bw in bw_contacts.keys() if bw not in fdr_sites]),
    })

print(f"✅ Analyzed {len(results)} structures")

# Statistical analysis
df = pd.DataFrame(results)
print("\n📈 Results summary:")
print(df[["entry_name", "coupling", "total_contacts", "bw_contacts", "fdr_bw_contacts"]])

# Statistical test
total_fdr_contacts = df["fdr_bw_contacts"].sum()
total_non_fdr_contacts = df["non_fdr_bw_contacts"].sum()
n_fdr_sites = len(fdr_sites)
n_non_fdr_sites = len(GP_CONTACT_SITES) - n_fdr_sites

print(f"\n🔍 Contact analysis:")
print(f"FDR BW sites: {total_fdr_contacts} contacts from {n_fdr_sites} sites")
print(f"Non-FDR BW sites: {total_non_fdr_contacts} contacts from {n_non_fdr_sites} sites")

# Fisher's exact test
contingency = [
    [total_fdr_contacts, n_fdr_sites * len(df) - total_fdr_contacts],
    [total_non_fdr_contacts, n_non_fdr_sites * len(df) - total_non_fdr_contacts]
]

oddsratio, pvalue = stats.fisher_exact(contingency, alternative='greater')
print(f"\n📊 Fisher's exact test:")
print(f"Odds ratio: {oddsratio:.3f}")
print(f"P-value: {pvalue:.2e}")

if pvalue < 0.05:
    print("✅ FDR-significant BW sites show significantly higher contact frequency!")
else:
    print("❌ No significant enrichment of FDR sites in predicted contacts")

# Visualization
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('AlphaFold-Multimer Validation: BW Site Contact Analysis', fontsize=14, fontweight='bold')

# 1. Contact frequency by coupling type
ax1 = axes[0, 0]
coupling_contacts = df.groupby("coupling")["bw_contacts"].mean()
sns.barplot(x=coupling_contacts.index, y=coupling_contacts.values, ax=ax1)
ax1.set_title('BW Contacts by G Protein Coupling')
ax1.set_ylabel('Mean BW Contacts')
ax1.tick_params(axis='x', rotation=45)

# 2. FDR vs non-FDR enrichment
ax2 = axes[0, 1]
contact_types = ['FDR Sites', 'Non-FDR Sites']
contact_counts = [total_fdr_contacts, total_non_fdr_contacts]
site_counts = [n_fdr_sites, n_non_fdr_sites]

x = np.arange(len(contact_types))
width = 0.35

ax2.bar(x - width/2, contact_counts, width, label='Observed Contacts', alpha=0.7)
ax2.bar(x + width/2, site_counts, width, label='Available Sites', alpha=0.7)
ax2.set_xlabel('BW Site Type')
ax2.set_ylabel('Count')
ax2.set_title(f'FDR Enrichment (p={pvalue:.2e})')
ax2.set_xticks(x)
ax2.set_xticklabels(contact_types)
ax2.legend()

# 3. Individual receptor results
ax3 = axes[1, 0]
colors = df["coupling"].map({'Gq': 'red', 'Gi': 'blue', 'Gs': 'green', 'G12': 'purple', 'Unknown': 'gray'})
ax3.scatter(range(len(df)), df["bw_contacts"], c=colors, s=100, alpha=0.7)
ax3.set_xlabel('Receptor Index')
ax3.set_ylabel('BW Contacts')
ax3.set_title('BW Contacts per Receptor')
ax3.grid(True, alpha=0.3)

# Add legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Gq'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Gi'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Gs'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='G12')]
ax3.legend(handles=legend_elements, loc='upper right')

# 4. Contact heatmap
ax4 = axes[1, 1]
if len(df) > 5:
    all_bw_sites = sorted(set().union(*[r['bw_contact_sites'] for r in results]))
    contact_matrix = np.zeros((len(df), len(all_bw_sites)))
    
    for i, result in enumerate(results):
        for j, bw_site in enumerate(all_bw_sites):
            if bw_site in result['bw_contact_sites']:
                contact_matrix[i, j] = 1
    
    sns.heatmap(contact_matrix, 
                xticklabels=all_bw_sites, 
                yticklabels=[r['entry_name'].split('_')[0].upper() for r in results],
                cmap='Reds', ax=ax4, cbar_kws={'label': 'Contact'})
    ax4.set_title('BW Site Contact Pattern')
    ax4.set_xlabel('BW Position')
    ax4.set_ylabel('Receptor')
else:
    ax4.text(0.5, 0.5, 'Insufficient data\nfor heatmap', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('BW Site Contact Pattern')

plt.tight_layout()
plt.savefig('/content/results/alphafold_validation/fig_af_validation.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Visualization saved to: /content/results/alphafold_validation/fig_af_validation.png")

# Save results
results_file = Path("/content/results/alphafold_validation/af_validation_results.csv")
df.to_csv(results_file, index=False)
print(f"💾 Results saved: {results_file}")

summary = {
    "total_receptors": len(df),
    "fdr_sites": n_fdr_sites,
    "non_fdr_sites": n_non_fdr_sites,
    "total_fdr_contacts": int(total_fdr_contacts),
    "total_non_fdr_contacts": int(total_non_fdr_contacts),
    "odds_ratio": float(oddsratio),
    "p_value": float(pvalue),
    "significant_enrichment": pvalue < 0.05
}

summary_file = Path("/content/results/alphafold_validation/af_validation_summary.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"💾 Summary saved: {summary_file}")

print(f"\n🎉 AlphaFold-Multimer validation complete!")
print(f"📊 Key finding: {'Significant' if pvalue < 0.05 else 'No'} enrichment of FDR sites (p={pvalue:.2e})")

# Create download package
print("\n📦 Creating download package...")
zip_path = "/content/af_multimer_results.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    # Add results
    for file in Path("/content/results/alphafold_validation").glob("*"):
        if file.is_file():
            zipf.write(file, file.name)
    
    # Add PDB structures
    for pdb_file in Path("/content/results/alphafold_validation/af_predictions").glob("*.pdb"):
        zipf.write(pdb_file, f"pdb_structures/{pdb_file.name}")

print(f"📦 Results packaged: {zip_path}")
print(f"📊 Size: {Path(zip_path).stat().st_size / 1024 / 1024:.1f} MB")

# Download
from google.colab import files
files.download(zip_path)

print("\n✅ Ready for manuscript integration!")
