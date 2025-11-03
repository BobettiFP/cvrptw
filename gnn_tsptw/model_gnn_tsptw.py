# cvrptw/POMO+PIP/gnn_tsptw/model_gnn_tsptw.py
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv

class MLP(nn.Module):
    def __init__(self, in_dim, hid, out_dim, act=True):
        super().__init__()
        layers = [nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, out_dim)]
        if act: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class TSPTWFeasGIN(nn.Module):
    """
    가장 단순한 백본:
    - node_in: [x,y,l,u] → hid
    - edge는 메시지패싱 시 GIN으로 집계(여기선 edge_attr 직접 사용 X: '제일 단순')
    - 로짓: 노드별 1 차원
    - 현재 노드/시간은 넣지 않는다(논문 기준 node feature만).
    """
    def __init__(self, in_dim=4, hid=128, layers=3):
        super().__init__()
        self.node_in = nn.Linear(in_dim, hid)
        self.convs = nn.ModuleList(
            [GINConv(MLP(hid, hid, hid)) for _ in range(layers)]
        )
        self.out = nn.Linear(hid, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.node_in(x)
        for conv in self.convs:
            x = conv(x, edge_index)
        logits = self.out(x).squeeze(-1)  # (N,)
        return logits
