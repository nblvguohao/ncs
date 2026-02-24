"""
Ballesteros–Weinstein (BW) site physicochemical features.

145-dimensional feature vector: 5 binary encodings × 29 G protein contact sites.
"""
import os
import json
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# 29 established G protein contact positions
GP_CONTACT_SITES = [
    "34.50", "34.51", "34.52", "34.53", "34.54", "34.55", "34.56", "34.57",
    "3.49", "3.50", "3.51", "3.53", "3.54", "3.55", "3.56",
    "5.61", "5.64", "5.65", "5.67", "5.68", "5.69", "5.71",
    "6.32", "6.33", "6.36", "6.37",
    "8.47", "8.48", "8.49",
]

# Physicochemical binary encodings
POSITIVE_AA = set("KRH")
NEGATIVE_AA = set("DE")
HYDROPHOBIC_AA = set("AILMFVPW")
AROMATIC_AA = set("FYW")

ENCODING_NAMES = ["is_positive", "is_negative", "is_hydrophobic", "is_aromatic", "is_gap"]


def _encode_residue(aa):
    """Encode a single amino acid as 5-dimensional binary vector."""
    if aa is None or aa == "-" or aa == "":
        return [0, 0, 0, 0, 1]  # gap
    return [
        1 if aa in POSITIVE_AA else 0,
        1 if aa in NEGATIVE_AA else 0,
        1 if aa in HYDROPHOBIC_AA else 0,
        1 if aa in AROMATIC_AA else 0,
        0,  # not a gap
    ]


def load_bw_cache(cache_path=None):
    """Load cached BW annotations from GPCRdb."""
    if cache_path is None:
        cache_path = os.path.join(DATA_DIR, "gpcrdb_residues_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_bw_residue(residue_data, bw_label):
    """Extract amino acid at a specific BW position from GPCRdb residue list."""
    for res in residue_data:
        display = res.get("display_generic_number") or res.get("generic_number", "")
        if isinstance(display, dict):
            display = display.get("label", "")
        # Normalize: GPCRdb may return "3.53x53" or "34.50x50"
        clean = str(display).split("x")[0].strip()
        if clean == bw_label:
            return res.get("amino_acid", res.get("aa", None))
    return None


def extract_bw_features(entry_name, bw_cache, sites=None):
    """Extract BW-site physicochemical features for one receptor.

    Parameters
    ----------
    entry_name : str
        GPCRdb entry name (e.g. '5ht2a_human').
    bw_cache : dict
        Cached BW residue annotations keyed by entry_name.
    sites : list of str or None
        BW positions to use. Defaults to GP_CONTACT_SITES.

    Returns
    -------
    np.ndarray of shape (n_sites * 5,)
    """
    if sites is None:
        sites = GP_CONTACT_SITES
    residues = bw_cache.get(entry_name, [])
    vec = []
    for bw in sites:
        aa = get_bw_residue(residues, bw)
        vec.extend(_encode_residue(aa))
    return np.array(vec, dtype=np.float32)


def build_bw_feature_matrix(entry_names, bw_cache, sites=None):
    """Build BW-site feature matrix for a list of receptors.

    Parameters
    ----------
    entry_names : list of str
    bw_cache : dict
    sites : list of str or None

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_sites * 5)
    feature_names : list of str
    """
    if sites is None:
        sites = GP_CONTACT_SITES
    feature_names = [f"{bw}_{enc}" for bw in sites for enc in ENCODING_NAMES]
    rows = [extract_bw_features(name, bw_cache, sites) for name in entry_names]
    X = np.vstack(rows)
    return X, feature_names
