import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.model_zoo.chem.gnn import GCNLayer
from dgl.batched_graph import max_nodes
from dgl.nn.pytorch import WeightAndSum
class GCN(nn.Module):
    def __init__(self, n_tasks=1, in_feats=16, gcn_hidden_feats=[256,256,256],
                 dropout=0.0):
        super(GCN, self).__init__()
        self.gnn_layers = nn.ModuleList()
        for i in range(len(gcn_hidden_feats)):
            out_feats = gcn_hidden_feats[i]
            self.gnn_layers.append(GCNLayer(in_feats, out_feats))
            in_feats = out_feats
        
        self.weighted_sum_readout = WeightAndSum(gcn_hidden_feats[-1])
        
        self.predictor = nn.Sequential(
            nn.Linear(gcn_hidden_feats[-1]*2, int(gcn_hidden_feats[-1])),
            nn.ReLU(),
            nn.Linear(int(gcn_hidden_feats[-1]), n_tasks)
        )
        self.encoder = nn.ModuleList([
            self.gnn_layers,
            self.weighted_sum_readout
        ])
        
    def forward(self, g):
        feats = g.ndata['h']
        h_g, feats = self.forward_encoder(g, feats)
        h = self.predictor(h_g)
        return h_g, h
    def forward_encoder(self, g, feats):
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        h_g_sum = self.weighted_sum_readout(g, feats)

        with g.local_scope():
            g.ndata['h'] = feats
            h_g_max = max_nodes(g, 'h')

        h_g = torch.cat([h_g_sum, h_g_max], dim=1)

        return h_g, feats