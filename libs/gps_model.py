import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from libs.layers import PMALayer
from libs.gps_layers import RWSELinear, GPSLayer, compute_rwse_batched


class GPSModel(nn.Module):
    """GPS (General, Powerful, Scalable) graph transformer for regression.

    Matches the MyModel interface exactly: forward() returns (prediction, alpha)
    so it can be dropped into any training loop that uses MyModel.

    Architecture per layer:
        node embedding → (optional RWSE) → N x GPSLayer → PMA/mean readout → linear

    Each GPSLayer = local GIN/GCN + global Transformer attention + FFN + LayerNorm.
    """
    def __init__(
            self,
            num_layers=4,
            hidden_dim=128,
            num_heads=4,
            dropout_prob=0.2,
            out_dim=1,
            readout='pma',
            initial_node_dim=58,
            initial_edge_dim=6,
            apply_sigmoid=False,
            norm_features=False,
            local_mp_type='gin',
            rwse_k=16,
        ):
        super().__init__()

        self.readout = readout
        self.num_layers = num_layers
        self.rwse_k = rwse_k
        self.use_rwse = (rwse_k > 0)

        self.embedding_node = nn.Linear(initial_node_dim, hidden_dim, bias=False)
        self.embedding_edge = nn.Linear(initial_edge_dim, hidden_dim, bias=False)

        if self.use_rwse:
            self.rwse_encoder = RWSELinear(rwse_k, hidden_dim)

        self.gps_layers = nn.ModuleList([
            GPSLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_prob=dropout_prob,
                local_mp_type=local_mp_type,
            )
            for _ in range(num_layers)
        ])

        if self.readout == 'pma':
            self.pma = PMALayer(
                k=1,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                norm_features=norm_features,
            )

        self.linear_out = nn.Linear(hidden_dim, out_dim, bias=True)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, graph, training=False):
        h = self.embedding_node(graph.ndata['h'].float())
        e_ij = self.embedding_edge(graph.edata['e_ij'].float())

        if self.use_rwse:
            # Compute topology-based encoding before storing embedded features
            # (compute_rwse_batched only reads graph topology, not node features)
            rwse = compute_rwse_batched(graph, self.rwse_k)
            h = h + self.rwse_encoder(rwse)

        graph.ndata['h'] = h
        graph.edata['e_ij'] = e_ij

        for layer in self.gps_layers:
            graph = layer(graph, training=training)

        alpha = None
        if self.readout in ['sum', 'mean', 'max']:
            out = dgl.readout_nodes(graph, 'h', op=self.readout)
        elif self.readout == 'pma':
            out, alpha = self.pma(graph)

        out = self.linear_out(out)

        if self.apply_sigmoid:
            out = torch.sigmoid(out)

        return out, alpha
