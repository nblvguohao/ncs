"""
Project configuration for GPCR SE(3)-GNN coupling prediction.
"""
import os

# ================================================================
# Paths
# ================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STRUCTURE_DIR = os.path.join(DATA_DIR, "structures")
EMBEDDING_DIR = os.path.join(DATA_DIR, "embeddings")
GRAPH_DIR = os.path.join(DATA_DIR, "graphs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ================================================================
# GPCRdb API
# ================================================================
GPCRDB_BASE_URL = "https://gpcrdb.org/services"

# Target species for multi-species expansion
TARGET_SPECIES = [
    "Homo sapiens",        # Human — primary
    "Mus musculus",        # Mouse
    "Rattus norvegicus",   # Rat
    "Bos taurus",          # Cow
    "Danio rerio",         # Zebrafish — evolutionary distance
    "Gallus gallus",       # Chicken
]

# GPCR families to include (exclude olfactory + taste type 2)
TARGET_FAMILIES = [
    # Class A
    "001_001",  # Aminergic
    "001_002",  # Peptide
    "001_003",  # Protein
    "001_004",  # Lipid
    "001_005",  # Melatonin
    "001_006",  # Nucleotide
    "001_007",  # Steroid
    "001_008",  # Alicarboxylic acid
    "001_009",  # Sensory (opsins)
    # Class B1
    "002_001",  # Secretin
    # Class C
    "004_001",  # Ion (CaSR)
    "004_002",  # Amino acid (mGlu, GABA_B)
    # Class F (Frizzled) — add for completeness
    "005_001",  # Frizzled
]

# ================================================================
# G protein coupling — 4-way classification
# ================================================================
G_PROTEIN_FAMILIES = ["Gs", "Gi/o", "Gq/11", "G12/13"]

# Coupling label encoding
COUPLING_LEVELS = {
    "primary": 2,
    "secondary": 1,
    "none": 0,
}

# ================================================================
# AlphaFold2 structure download
# ================================================================
AF2_API_URL = "https://alphafold.ebi.ac.uk/api/prediction"
AF2_FILE_URL = "https://alphafold.ebi.ac.uk/files"

# ================================================================
# ESM-2 model
# ================================================================
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"  # 650M params — much larger for GNN
ESM2_REPR_LAYER = 33  # Last layer
ESM2_BATCH_SIZE = 2   # RTX 4060 8GB VRAM

# ================================================================
# 3D Graph construction
# ================================================================
CONTACT_RADIUS = 10.0      # Angstroms — C-alpha contact radius for edges
MAX_SEQ_LENGTH = 1200      # Truncate very long sequences (Class C)
BW_CONTACT_SITES = [       # 29 known G protein contact BW positions
    "2.39", "2.40", "3.46", "3.49", "3.50", "3.51", "3.52", "3.53", "3.54",
    "3.55", "3.56", "34.50", "34.51", "34.52", "34.53", "34.54", "34.55",
    "34.56", "34.57", "5.61", "5.64", "5.65", "5.67", "5.68", "5.69",
    "5.71", "5.72", "5.74", "5.75",
]

# ================================================================
# SE(3) GNN model
# ================================================================
GNN_CONFIG = {
    "node_feat_dim": 1280,     # ESM-2 650M embedding dim
    "edge_feat_dim": 16,       # Edge features (distance bins, angle features)
    "hidden_dim": 64,          # Reduced for 8GB VRAM (e3nn TP scales quadratically)
    "num_layers": 3,
    "num_heads": 4,
    "lmax": 1,                 # Reduced: 0e + 1o only (saves memory vs lmax=2)
    "dropout": 0.1,
    "pool": "attention",
}

# ================================================================
# Contrastive learning
# ================================================================
CONTRASTIVE_CONFIG = {
    "projection_dim": 64,      # Contrastive projection head output dim
    "temperature": 0.07,       # NT-Xent temperature
    "lambda_cls": 1.0,         # Classification loss weight
    "lambda_contrast": 0.5,    # Contrastive loss weight
    "augmentation": ["mask_nodes", "perturb_coords"],  # Graph augmentations
    "mask_ratio": 0.15,
}

# ================================================================
# Training
# ================================================================
TRAIN_CONFIG = {
    "batch_size": 4,           # Small batch for 8GB VRAM with large graphs
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 200,
    "patience": 30,            # Early stopping
    "scheduler": "cosine",
    "warmup_epochs": 10,
    "num_workers": 4,
    "seed": 42,
}

# ================================================================
# Zero-shot evaluation
# ================================================================
ZEROSHOT_CONFIG = {
    "leave_out": "subfamily",      # Leave-one-subfamily-out
    "n_folds": None,               # Auto-determined by number of subfamilies
    "metrics": ["auroc", "auprc", "f1", "mcc"],
}
