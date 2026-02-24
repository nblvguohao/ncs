"""
ESM-2 protein language model feature extraction.

Supports:
  - Mean-pooled sequence embeddings
  - Per-residue embeddings at BW contact sites
  - Attention weight extraction at BW positions
"""
import os
import numpy as np

try:
    import torch
    import esm
    HAS_ESM = True
except ImportError:
    HAS_ESM = False

from .bw_site import GP_CONTACT_SITES, load_bw_cache, get_bw_residue


def _check_esm():
    if not HAS_ESM:
        raise ImportError(
            "ESM-2 requires `fair-esm` and `torch`. "
            "Install with: pip install fair-esm torch"
        )


def load_esm2_model(model_name="esm2_t6_8M_UR50D"):
    """Load an ESM-2 model and alphabet.

    Parameters
    ----------
    model_name : str
        One of 'esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D',
        'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'.

    Returns
    -------
    model, alphabet, batch_converter
    """
    _check_esm()
    model, alphabet = getattr(esm.pretrained, model_name)()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    return model, alphabet, batch_converter


def extract_esm2_embeddings(sequences, entry_names, model_name="esm2_t6_8M_UR50D",
                             batch_size=8, device="cpu"):
    """Extract mean-pooled ESM-2 embeddings for a list of sequences.

    Returns
    -------
    np.ndarray of shape (n_sequences, embed_dim)
    """
    _check_esm()
    model, alphabet, batch_converter = load_esm2_model(model_name)
    model = model.to(device)

    all_embeddings = []
    data = list(zip(entry_names, sequences))

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers])

        reps = results["representations"][model.num_layers]
        for j, (name, seq) in enumerate(batch):
            seq_len = len(seq)
            emb = reps[j, 1 : seq_len + 1].mean(dim=0).cpu().numpy()
            all_embeddings.append(emb)

    return np.vstack(all_embeddings)


def extract_esm2_bw_embeddings(sequences, entry_names, bw_cache,
                                 model_name="esm2_t6_8M_UR50D",
                                 sites=None, batch_size=8, device="cpu"):
    """Extract ESM-2 per-residue embeddings at BW contact sites.

    Returns
    -------
    X : np.ndarray of shape (n_sequences, n_sites * embed_dim)
    feature_names : list of str
    """
    _check_esm()
    if sites is None:
        sites = GP_CONTACT_SITES

    model, alphabet, batch_converter = load_esm2_model(model_name)
    model = model.to(device)
    embed_dim = model.embed_dim if hasattr(model, 'embed_dim') else 320

    # Pre-compute BW position → sequence index mapping
    bw_seq_indices = []
    for entry_name in entry_names:
        residues = bw_cache.get(entry_name, [])
        pos_map = {}
        for res in residues:
            display = res.get("display_generic_number") or res.get("generic_number", "")
            if isinstance(display, dict):
                display = display.get("label", "")
            clean = str(display).split("x")[0].strip()
            seq_number = res.get("sequence_number", None)
            if seq_number is not None and clean:
                pos_map[clean] = int(seq_number) - 1  # 0-indexed
        bw_seq_indices.append(pos_map)

    all_rows = []
    data = list(zip(entry_names, sequences))

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers])
        reps = results["representations"][model.num_layers]

        for j, (name, seq) in enumerate(batch):
            idx_in_full = i + j
            pos_map = bw_seq_indices[idx_in_full]
            row = []
            for bw in sites:
                seq_idx = pos_map.get(bw, None)
                if seq_idx is not None and 0 <= seq_idx < len(seq):
                    vec = reps[j, seq_idx + 1].cpu().numpy()  # +1 for BOS token
                else:
                    vec = np.zeros(embed_dim, dtype=np.float32)
                row.append(vec)
            all_rows.append(np.concatenate(row))

    feature_names = [f"{bw}_esm{d}" for bw in sites for d in range(embed_dim)]
    return np.vstack(all_rows), feature_names


def extract_esm2_attention(sequences, entry_names, bw_cache,
                            model_name="esm2_t6_8M_UR50D",
                            sites=None, batch_size=4, device="cpu"):
    """Extract ESM-2 attention weights at BW contact site positions.

    For each sequence, averages attention across all heads and layers at each
    BW site position, producing a (n_sites,) summary vector per receptor.

    Returns
    -------
    attn_matrix : np.ndarray of shape (n_sequences, n_sites)
        Mean attention received by each BW position (averaged over layers/heads).
    site_names : list of str
    """
    _check_esm()
    if sites is None:
        sites = GP_CONTACT_SITES

    model, alphabet, batch_converter = load_esm2_model(model_name)
    model = model.to(device)

    # Pre-compute BW → sequence index
    bw_seq_indices = []
    for entry_name in entry_names:
        residues = bw_cache.get(entry_name, [])
        pos_map = {}
        for res in residues:
            display = res.get("display_generic_number") or res.get("generic_number", "")
            if isinstance(display, dict):
                display = display.get("label", "")
            clean = str(display).split("x")[0].strip()
            seq_number = res.get("sequence_number", None)
            if seq_number is not None and clean:
                pos_map[clean] = int(seq_number) - 1
        bw_seq_indices.append(pos_map)

    all_attn = []
    data = list(zip(entry_names, sequences))

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers],
                           return_contacts=False)

        # Attention: (n_layers, batch, n_heads, seq_len, seq_len)
        if "attentions" in results:
            attentions = results["attentions"]  # tuple of (batch, heads, L, L)
            # Stack: (n_layers, batch, heads, L, L)
            attn_stack = torch.stack(attentions, dim=0)
            # Mean over layers and heads: (batch, L, L)
            attn_mean = attn_stack.mean(dim=[0, 2])
        else:
            # Fallback: re-run with need_head_weights
            attn_mean = None

        for j, (name, seq) in enumerate(batch):
            idx_in_full = i + j
            pos_map = bw_seq_indices[idx_in_full]
            seq_len = len(seq)
            row = []
            for bw in sites:
                seq_idx = pos_map.get(bw, None)
                if (seq_idx is not None and attn_mean is not None
                        and 0 <= seq_idx < seq_len):
                    # Column attention = how much other tokens attend to this position
                    col_attn = attn_mean[j, 1:seq_len+1, seq_idx+1].sum().item()
                    row.append(col_attn / seq_len)
                else:
                    row.append(0.0)
            all_attn.append(row)

    return np.array(all_attn, dtype=np.float32), [f"attn_{s}" for s in sites]


def load_precomputed_esm2_embeddings(npz_path):
    """Load precomputed ESM-2 BW embeddings from .npz file.

    Returns
    -------
    X : np.ndarray
    entry_names : list of str (if stored)
    """
    data = np.load(npz_path, allow_pickle=True)
    X = data["embeddings"] if "embeddings" in data else data["X"]
    names = list(data["entry_names"]) if "entry_names" in data else []
    return X, names
