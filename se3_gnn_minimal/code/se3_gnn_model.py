#!/usr/bin/env python3
"""
Phase 3: SE(3)-Equivariant Graph Neural Network for GPCR coupling prediction.

Architecture:
  1. Node encoder: Linear projection of ESM-2 embeddings + pLDDT
  2. SE(3) equivariant message passing layers (e3nn-based)
  3. Invariant graph-level pooling (attention-weighted)
  4. Dual head: classification + contrastive projection

The SE(3) equivariance ensures predictions are invariant to rotation/translation
of the 3D structure, which is physically meaningful for protein function.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from e3nn import o3
    from e3nn.nn import FullyConnectedNet, Gate
    from e3nn.o3 import FullyConnectedTensorProduct, Irreps
    HAS_E3NN = True
except ImportError:
    HAS_E3NN = False
    print("WARNING: e3nn not installed. Using invariant GNN fallback.")

try:
    from torch_geometric.nn import global_mean_pool, global_add_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ================================================================
# Spherical harmonics edge encoder
# ================================================================
class RadialBasisEncoding(nn.Module):
    """Gaussian radial basis functions for distance encoding."""

    def __init__(self, num_basis=16, cutoff=10.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        # Evenly spaced centers
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        self.width = 0.5 * (cutoff / num_basis) ** 2

    def forward(self, dist):
        """dist: (E,) -> (E, num_basis)"""
        dist = dist.unsqueeze(-1)  # (E, 1)
        return torch.exp(-self.width * (dist - self.centers) ** 2)


class EdgeEncoder(nn.Module):
    """
    Encode edge features: radial basis of distance + spherical harmonics of direction.
    """

    def __init__(self, num_radial=16, cutoff=10.0, lmax=2):
        super().__init__()
        self.radial_basis = RadialBasisEncoding(num_radial, cutoff)
        self.lmax = lmax
        # Spherical harmonics irreps: 1x0e + 1x1o + 1x2e (for lmax=2)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax) if HAS_E3NN else None

    def forward(self, pos, edge_index):
        """
        pos: (N, 3)
        edge_index: (2, E)
        Returns: edge_sh (E, sh_dim), edge_radial (E, num_radial), edge_dist (E,)
        """
        src, dst = edge_index
        rel_pos = pos[dst] - pos[src]  # (E, 3)
        dist = rel_pos.norm(dim=-1, keepdim=False)  # (E,)

        # Radial basis
        edge_radial = self.radial_basis(dist)  # (E, num_radial)

        # Spherical harmonics (SE(3) equivariant)
        if HAS_E3NN:
            rel_pos_normalized = F.normalize(rel_pos, dim=-1)
            edge_sh = o3.spherical_harmonics(
                self.sh_irreps, rel_pos_normalized, normalize=True, normalization="component"
            )  # (E, sh_dim)
        else:
            edge_sh = F.normalize(rel_pos, dim=-1)  # Fallback: unit direction

        return edge_sh, edge_radial, dist


# ================================================================
# SE(3) Equivariant Convolution Layer
# ================================================================
class SE3ConvLayer(nn.Module):
    """
    A single SE(3)-equivariant message passing layer.

    Uses tensor products of node features (scalar) with edge spherical harmonics
    to produce equivariant messages, weighted by learned radial functions.
    """

    def __init__(self, irreps_in, irreps_out, irreps_sh, num_radial=16, hidden_dim=64):
        super().__init__()
        if not HAS_E3NN:
            # Fallback: simple message passing
            in_dim = int(irreps_in.split("x")[0]) if isinstance(irreps_in, str) else 256
            out_dim = int(irreps_out.split("x")[0]) if isinstance(irreps_out, str) else 256
            self.fallback = True
            self.linear = nn.Linear(in_dim + 3 + num_radial, out_dim)
            self.norm = nn.LayerNorm(out_dim)
            return

        self.fallback = False
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)

        # Tensor product for message computation
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False
        )

        # Radial network: maps radial basis to tensor product weights
        self.radial_net = FullyConnectedNet(
            [num_radial, hidden_dim, self.tp.weight_numel],
            act=torch.nn.functional.silu,
        )

        # Self-interaction
        self.self_interaction = o3.Linear(self.irreps_in, self.irreps_out)

        # Layer norm (only on scalar components)
        scalar_dim = sum(mul * (2 * l + 1) for mul, (l, p) in self.irreps_out if l == 0)
        self.norm = nn.LayerNorm(scalar_dim) if scalar_dim > 0 else None

    def forward(self, node_feat, edge_index, edge_sh, edge_radial):
        """
        node_feat: (N, feat_dim)
        edge_index: (2, E)
        edge_sh: (E, sh_dim)
        edge_radial: (E, num_radial)
        """
        if self.fallback:
            src, dst = edge_index
            msg = torch.cat([node_feat[src], edge_sh, edge_radial], dim=-1)
            msg = self.linear(msg)
            # Aggregate
            out = torch.zeros_like(node_feat[:, :msg.shape[-1]])
            out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
            return self.norm(F.silu(out))

        src, dst = edge_index
        num_nodes = node_feat.shape[0]

        # Compute TP weights from radial basis
        tp_weights = self.radial_net(edge_radial)  # (E, weight_numel)

        # Message: tensor product of source features with edge SH
        messages = self.tp(node_feat[src], edge_sh, tp_weights)  # (E, out_dim)

        # Aggregate messages (sum)
        out = torch.zeros(num_nodes, messages.shape[-1],
                          device=messages.device, dtype=messages.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Self-interaction (skip connection)
        out = out + self.self_interaction(node_feat)

        return out


# ================================================================
# Full SE(3) GNN Model
# ================================================================
class SE3GPCRGNN(nn.Module):
    """
    SE(3)-equivariant GNN for GPCR G-protein coupling prediction.

    Architecture:
      Input -> Node Encoder -> [SE3Conv + Gate] × L -> Invariant Pool -> Heads
    """

    def __init__(
        self,
        node_input_dim=1281,   # ESM-2 (1280) + pLDDT (1)
        hidden_dim=256,
        num_layers=4,
        lmax=2,
        num_radial=16,
        cutoff=10.0,
        num_classes=4,         # Gs, Gi/o, Gq/11, G12/13
        projection_dim=128,    # Contrastive head output
        dropout=0.1,
        pool_type="attention",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pool_type = pool_type

        # Edge encoder
        self.edge_encoder = EdgeEncoder(num_radial, cutoff, lmax)

        # Node input projection (invariant: scalar only)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Define irreps for SE(3) layers
        if HAS_E3NN:
            self.irreps_hidden = o3.Irreps(f"{hidden_dim}x0e")  # Scalar features
            sh_irreps = o3.Irreps.spherical_harmonics(lmax)

            self.conv_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.conv_layers.append(
                    SE3ConvLayer(
                        irreps_in=self.irreps_hidden,
                        irreps_out=self.irreps_hidden,
                        irreps_sh=sh_irreps,
                        num_radial=num_radial,
                        hidden_dim=hidden_dim,
                    )
                )
        else:
            self.conv_layers = nn.ModuleList([
                SE3ConvLayer(
                    irreps_in=str(hidden_dim),
                    irreps_out=str(hidden_dim),
                    irreps_sh="3",
                    num_radial=num_radial,
                    hidden_dim=hidden_dim,
                ) for _ in range(num_layers)
            ])

        # Gate activations between layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Attention pooling
        if pool_type == "attention":
            self.pool_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1),
            )

        # Classification head (multi-label: 4 G proteins)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Contrastive projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def encode_graph(self, x, pos, edge_index, batch=None):
        """
        Encode a graph into a fixed-size representation.

        Args:
            x: (N, node_input_dim) node features
            pos: (N, 3) coordinates
            edge_index: (2, E) edge indices
            batch: (N,) batch assignment for batched graphs

        Returns:
            graph_emb: (B, hidden_dim) graph-level embedding
            node_emb: (N, hidden_dim) final node embeddings
        """
        # Encode edges
        edge_sh, edge_radial, edge_dist = self.edge_encoder(pos, edge_index)

        # Node input projection
        h = self.node_encoder(x)  # (N, hidden_dim)

        # SE(3) message passing layers
        for i, conv in enumerate(self.conv_layers):
            h_new = conv(h, edge_index, edge_sh, edge_radial)
            h = self.layer_norms[i](h + h_new)  # Residual connection
            h = self.dropout(h)

        node_emb = h  # (N, hidden_dim)

        # Graph-level pooling
        if self.pool_type == "attention":
            gate_scores = self.pool_gate(h)  # (N, 1)
            if batch is not None:
                # Softmax within each graph
                gate_scores = self._batched_softmax(gate_scores, batch)
            else:
                gate_scores = torch.softmax(gate_scores, dim=0)
            graph_emb = (h * gate_scores).sum(dim=0, keepdim=True) if batch is None \
                else self._batched_weighted_sum(h, gate_scores, batch)
        else:
            if batch is not None:
                graph_emb = global_mean_pool(h, batch)
            else:
                graph_emb = h.mean(dim=0, keepdim=True)

        return graph_emb, node_emb

    def forward(self, x, pos, edge_index, batch=None):
        """
        Full forward pass.

        Returns:
            logits: (B, 4) coupling prediction logits
            projection: (B, projection_dim) contrastive embedding
            graph_emb: (B, hidden_dim) graph embedding
        """
        graph_emb, node_emb = self.encode_graph(x, pos, edge_index, batch)

        logits = self.classifier(graph_emb)
        projection = F.normalize(self.projector(graph_emb), dim=-1)

        return logits, projection, graph_emb

    def _batched_softmax(self, scores, batch):
        """Softmax over nodes within each graph in a batch."""
        # Subtract max for numerical stability
        max_scores = torch.zeros(batch.max() + 1, 1, device=scores.device)
        max_scores.scatter_reduce_(0, batch.unsqueeze(-1), scores, reduce="amax")
        scores = scores - max_scores[batch]
        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(batch.max() + 1, 1, device=scores.device)
        sum_exp.scatter_add_(0, batch.unsqueeze(-1), exp_scores)
        return exp_scores / (sum_exp[batch] + 1e-8)

    def _batched_weighted_sum(self, h, weights, batch):
        """Weighted sum pooling within each graph."""
        weighted = h * weights
        B = batch.max() + 1
        result = torch.zeros(B, h.shape[-1], device=h.device)
        result.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted), weighted)
        return result


# ================================================================
# Model factory
# ================================================================
def build_model(config=None):
    """Build SE3GPCRGNN from config dict."""
    if config is None:
        from configs.config import GNN_CONFIG
        config = GNN_CONFIG

    model = SE3GPCRGNN(
        node_input_dim=config.get("node_feat_dim", 1280) + 1,  # +1 for pLDDT
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 4),
        lmax=config.get("lmax", 2),
        num_radial=config.get("edge_feat_dim", 16),
        cutoff=10.0,
        num_classes=4,
        projection_dim=config.get("projection_dim", 128) if "projection_dim" in config else 128,
        dropout=config.get("dropout", 0.1),
        pool_type=config.get("pool", "attention"),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: SE3GPCRGNN")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  SE(3) equivariant: {HAS_E3NN}")

    return model


if __name__ == "__main__":
    # Quick test
    model = build_model()
    print(model)

    # Dummy forward pass
    N = 300  # residues
    x = torch.randn(N, 1281)
    pos = torch.randn(N, 3) * 20
    # Build edges (random for test)
    edge_index = torch.randint(0, N, (2, N * 10))

    logits, proj, emb = model(x, pos, edge_index)
    print(f"\nDummy forward pass:")
    print(f"  logits: {logits.shape}")       # (1, 4)
    print(f"  projection: {proj.shape}")     # (1, 128)
    print(f"  graph_emb: {emb.shape}")       # (1, 256)
