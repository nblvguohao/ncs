"""
Handcrafted sequence-level features for GPCR coupling prediction.

99-dimensional feature vector per receptor:
  - Amino acid composition (20)
  - Physicochemical ratios (6)
  - Conserved GPCR motifs (14)
  - TM helix count (1)
  - Intracellular loop features (14)
  - C-terminal features (4)
  - Regional features (9)
  - Sequence complexity (1)
  - Dipeptide frequencies (20)
  - G-alpha k-mer overlap (5)
  - GPCR class indicators (3)
"""
import re
import numpy as np
from collections import defaultdict
from ..data.dataset import G_PROTEIN_SEQS


def _find_tm_regions(sequence):
    """Predict transmembrane regions by hydrophobicity scanning."""
    hydro_set = set("AILMFVW")
    seq_len = len(sequence)
    tm_regions, in_tm, tm_start = [], False, 0
    for i in range(seq_len - 20):
        window = sequence[i : i + 20]
        hf = sum(1 for aa in window if aa in hydro_set) / 20
        if hf >= 0.55 and not in_tm:
            tm_start, in_tm = i, True
        elif hf < 0.35 and in_tm:
            tm_regions.append((tm_start, i))
            in_tm = False
    if in_tm:
        tm_regions.append((tm_start, seq_len))
    merged = []
    for s, e in tm_regions:
        if merged and s - merged[-1][1] < 10:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged[:10]


def _extract_icl_features(sequence, tm_regions):
    """Extract intracellular loop features."""
    features = {}
    hydrophobic = set("AILMFVPW")
    positive, negative, polar = set("KRH"), set("DE"), set("STNQ")
    icls = []
    for i in range(len(tm_regions) - 1):
        gs, ge = tm_regions[i][1], tm_regions[i + 1][0]
        if ge > gs:
            icls.append(sequence[gs:ge])
    for icl_idx, icl_name in [(1, "icl2"), (3, "icl3")]:
        if icl_idx < len(icls) and len(icls[icl_idx]) > 0:
            icl = icls[icl_idx]
            rl = len(icl)
            features[f"{icl_name}_len"] = rl
            features[f"{icl_name}_hydro"] = sum(1 for a in icl if a in hydrophobic) / rl
            features[f"{icl_name}_pos"] = sum(1 for a in icl if a in positive) / rl
            features[f"{icl_name}_neg"] = sum(1 for a in icl if a in negative) / rl
            features[f"{icl_name}_polar"] = sum(1 for a in icl if a in polar) / rl
            features[f"{icl_name}_charge"] = features[f"{icl_name}_pos"] - features[f"{icl_name}_neg"]
        else:
            for s in ["_len", "_hydro", "_pos", "_neg", "_polar", "_charge"]:
                features[f"{icl_name}{s}"] = 0
    all_icl = "".join(icls) if icls else ""
    if all_icl:
        rl = len(all_icl)
        features["icl_total_len"] = rl
        features["icl_total_charge"] = (
            sum(1 for a in all_icl if a in positive)
            - sum(1 for a in all_icl if a in negative)
        ) / rl
    else:
        features["icl_total_len"] = 0
        features["icl_total_charge"] = 0
    return features


def _get_kmer_set(seq, k=3):
    return set(seq[i : i + k] for i in range(len(seq) - k + 1))


def extract_handcrafted_features(sequence, family_slug=""):
    """Extract 99-dimensional handcrafted feature vector from a GPCR sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    family_slug : str
        GPCRdb family slug (e.g. '001_001_001_001').

    Returns
    -------
    dict
        Feature name -> float value.
    """
    features = {}
    seq_len = len(sequence)
    if seq_len == 0:
        return features

    features["length"] = seq_len
    features["log_length"] = np.log(seq_len)

    # Amino acid composition (20)
    aa_counts = defaultdict(int)
    for aa in sequence:
        aa_counts[aa] += 1
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        features[f"{aa}_ratio"] = aa_counts[aa] / seq_len

    # Physicochemical ratios (6)
    hydrophobic = set("AILMFVPW")
    polar, positive, negative, aromatic = set("STNQ"), set("KRH"), set("DE"), set("FYW")
    features["hydro_ratio"] = sum(aa_counts[a] for a in hydrophobic) / seq_len
    features["polar_ratio"] = sum(aa_counts[a] for a in polar) / seq_len
    features["pos_ratio"] = sum(aa_counts[a] for a in positive) / seq_len
    features["neg_ratio"] = sum(aa_counts[a] for a in negative) / seq_len
    features["arom_ratio"] = sum(aa_counts[a] for a in aromatic) / seq_len
    features["charge_ratio"] = features["pos_ratio"] - features["neg_ratio"]

    # Conserved GPCR motifs (14)
    features["has_DRY"] = 1.0 if "DRY" in sequence else 0.0
    features["has_ERY"] = 1.0 if "ERY" in sequence else 0.0
    features["has_NPxxY"] = 1.0 if re.search(r"NP..Y", sequence) else 0.0
    features["has_CWxP"] = 1.0 if re.search(r"CW.P", sequence) else 0.0
    features["has_QAKK"] = 1.0 if "QAKK" in sequence else 0.0
    features["has_PLAT"] = 1.0 if "PLAT" in sequence else 0.0
    features["has_HKKLR"] = 1.0 if "HKKLR" in sequence else 0.0
    features["has_DRYLV"] = 1.0 if re.search(r"DRY.V", sequence) else 0.0
    features["has_HEK"] = 1.0 if "HEK" in sequence else 0.0
    features["has_SLRT"] = 1.0 if "SLRT" in sequence else 0.0
    features["has_PMSNFR"] = 1.0 if "PMSNFR" in sequence else 0.0
    features["has_AAAQQ"] = 1.0 if "AAAQQ" in sequence else 0.0
    features["has_KKLRT"] = 1.0 if "KKLRT" in sequence else 0.0
    features["has_PMSN"] = 1.0 if "PMSN" in sequence else 0.0

    # TM helix count (1)
    tm_regions = _find_tm_regions(sequence)
    features["tm_count"] = len(tm_regions)

    # Intracellular loop features (14)
    features.update(_extract_icl_features(sequence, tm_regions))

    # C-terminal features (4)
    n_term = sequence[:50] if seq_len >= 50 else sequence
    c_term = sequence[-50:] if seq_len >= 50 else sequence
    mid = sequence[seq_len // 3 : 2 * seq_len // 3]
    for name, region in [("n", n_term), ("c", c_term), ("m", mid)]:
        rl = len(region)
        if rl > 0:
            features[f"{name}_hydro"] = sum(1 for a in region if a in hydrophobic) / rl
            features[f"{name}_charged"] = sum(1 for a in region if a in positive | negative) / rl
            features[f"{name}_arom"] = sum(1 for a in region if a in aromatic) / rl
    npxxy = list(re.finditer(r"NP..Y", sequence))
    if npxxy:
        c_tail = sequence[npxxy[-1].end() :]
        features["c_tail_len"] = len(c_tail)
        features["c_tail_ratio"] = len(c_tail) / seq_len
        features["c_tail_charged"] = (
            sum(1 for a in c_tail if a in positive | negative) / max(len(c_tail), 1)
        )
        features["c_tail_pos"] = sum(1 for a in c_tail if a in positive) / max(len(c_tail), 1)
    else:
        features["c_tail_len"] = features["c_tail_ratio"] = 0
        features["c_tail_charged"] = features["c_tail_pos"] = 0

    # Sequence complexity (1)
    features["complexity"] = len(set(sequence)) / 20.0

    # Dipeptide frequencies (20)
    dipeptides = defaultdict(int)
    for i in range(seq_len - 1):
        dipeptides[sequence[i : i + 2]] += 1
    for dp in [
        "LL", "VL", "LV", "II", "FF", "FL", "LF", "AI", "IA", "AL",
        "LA", "FI", "IF", "IV", "VI", "GV", "VG", "SS", "TT", "PP",
    ]:
        features[f"dp_{dp}"] = dipeptides[dp] / max(seq_len - 1, 1)

    # G-alpha k-mer overlap features (5)
    k = 3
    seq_kmers = _get_kmer_set(sequence, k)
    for gname in ("gnaq", "gnas", "gnai"):
        gk = _get_kmer_set(G_PROTEIN_SEQS[gname], k)
        features[f"{gname}_overlap"] = len(seq_kmers & gk) / len(seq_kmers) if seq_kmers else 0
    features["gq_vs_gs"] = features.get("gnaq_overlap", 0) - features.get("gnas_overlap", 0)
    features["gq_vs_gi"] = features.get("gnaq_overlap", 0) - features.get("gnai_overlap", 0)

    # GPCR class indicators (3)
    features["is_classA"] = 1.0 if family_slug.startswith("001") else 0.0
    features["is_classB"] = 1.0 if family_slug.startswith("002") else 0.0
    features["is_classC"] = 1.0 if family_slug.startswith("004") else 0.0

    return features


def build_feature_matrix(sequences, families=None):
    """Build feature matrix for a list of sequences.

    Parameters
    ----------
    sequences : list of str
    families : list of str or None

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    feature_names : list of str
    """
    if families is None:
        families = [""] * len(sequences)

    feature_names = None
    rows = []
    for seq, fam in zip(sequences, families):
        feat = extract_handcrafted_features(seq, fam)
        if feature_names is None:
            feature_names = sorted(feat.keys())
        rows.append([feat.get(n, 0) for n in feature_names])

    X = np.array(rows, dtype=np.float32)
    return X, feature_names
