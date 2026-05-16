import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from libs.layers import GraphIsomorphism, GraphConvolution, MLP


class RWSELinear(nn.Module):
    """Projects K-step random-walk landing probabilities into hidden_dim."""
    def __init__(self, rwse_k, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(rwse_k, hidden_dim, bias=False)

    def forward(self, rwse):
        return self.linear(rwse.float())


def compute_rwse_batched(graph, k):
    """Compute random-walk structural encoding for a batched DGL graph.

    For each node, computes the probability of landing back on itself after
    1, 2, ..., k random-walk steps. Returns a (total_N, k) float tensor.
    Isolated nodes (no edges) receive all-zero encodings.
    """
    device = graph.ndata['h'].device
    graphs = dgl.unbatch(graph)
    rwse_list = []
    for g in graphs:
        n = g.num_nodes()
        src, dst = g.edges()
        if len(src) == 0:
            rwse_list.append(torch.zeros(n, k, device=device))
            continue
        src = src.long()
        dst = dst.long()
        A = torch.zeros(n, n, device=device)
        A[src, dst] = 1.0
        D = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
        P = A / D          # row-stochastic transition matrix
        Pk = P.clone()
        rwse = torch.zeros(n, k, device=device)
        for step in range(k):
            rwse[:, step] = Pk.diagonal()
            if step < k - 1:
                Pk = Pk @ P
        rwse_list.append(rwse)
    return torch.cat(rwse_list, dim=0)


class GPSLayer(nn.Module):
    """GPS layer: local MP + global self-attention + FFN with pre-norm.

    Combines a local message-passing branch (GIN or GCN) with a global
    Transformer self-attention branch. Both branches operate on the same
    input features, and their outputs are summed then normalised.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout_prob=0.2, local_mp_type='gin'):
        super().__init__()

        if local_mp_type == 'gin':
            self.local_mp = GraphIsomorphism(
                hidden_dim=hidden_dim,
                dropout_prob=0.0,   # GPS-level dropout applied after combining
            )
        elif local_mp_type == 'gcn':
            self.local_mp = GraphConvolution(
                hidden_dim=hidden_dim,
                dropout_prob=0.0,
            )
        else:
            raise ValueError(f"local_mp_type must be 'gin' or 'gcn', got {local_mp_type}")

        # dropout=0.0 so the model.eval() flag does not silently disable
        # dropout during MC-Dropout inference; F.dropout below is used instead
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, 4 * hidden_dim, hidden_dim)
        self.prob = dropout_prob

    def forward(self, graph, training=False):
        h0 = graph.ndata['h']
        lengths = graph.batch_num_nodes().tolist()

        # --- Local MP branch ---
        # GraphIsomorphism/GraphConvolution update graph.ndata['h'] in-place
        graph = self.local_mp(graph, training=False)
        h_local = graph.ndata['h']   # includes the residual already

        # --- Global self-attention branch (operates on pre-update h0) ---
        chunks = torch.split(h0, lengths)
        padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True)  # (B, max_N, D)
        B, max_N, _ = padded.shape

        # True = padding position (will be ignored by attention)
        key_mask = torch.zeros(B, max_N, dtype=torch.bool, device=h0.device)
        for i, length in enumerate(lengths):
            if length < max_N:
                key_mask[i, length:] = True

        attn_out, _ = self.global_attn(
            padded, padded, padded,
            key_padding_mask=key_mask,
            need_weights=False,
        )
        # Unpad: collect only the valid (non-padding) rows
        h_global = torch.cat([attn_out[i, :lengths[i]] for i in range(B)], dim=0)
        h_global = F.dropout(h_global, p=self.prob, training=training)

        # --- Combine, normalise, FFN ---
        h = self.norm1(h_local + h_global)
        h = self.norm2(h + self.ffn(h))
        h = F.dropout(h, p=self.prob, training=training)

        graph.ndata['h'] = h
        return graph
