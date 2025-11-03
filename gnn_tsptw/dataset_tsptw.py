# cvrptw/POMO+PIP/gnn_tsptw/dataset_tsptw.py
import json, math, random
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch_geometric.data import Data

Tensor = torch.Tensor

def _euclid(a: Tensor, b: Tensor) -> Tensor:
    return torch.norm(a - b, dim=-1)

def build_step_graph(coords: Tensor, windows: Tensor,
                     current_node: int, current_time: float, visited: Tensor) -> Data:
    """
    coords: (N,2) float
    windows: (N,2) float [l_i, u_i]
    current_node: int
    current_time: float
    visited: (N,) bool
    """
    N = coords.size(0)
    # node features: [x, y, l, u]
    x = torch.cat([coords, windows], dim=1)  # (N,4)

    # complete graph
    src = torch.arange(N).repeat_interleave(N)
    dst = torch.arange(N).repeat(N)
    edge_index = torch.stack([src, dst], dim=0)  # (2, N^2)

    # edge attr = Euclidean distance
    diffs = coords[src] - coords[dst]
    edge_attr = torch.norm(diffs, dim=1, keepdim=True)

    # arrival from current
    dist_from_cur = _euclid(coords, coords[current_node].unsqueeze(0))
    arrival = current_time + dist_from_cur  # (N,)

    l, u = windows[:, 0], windows[:, 1]
    feasible = (~visited) & (arrival <= u)

    y = feasible.float()
    mask = (~visited & (arrival <= u)).float()  # 학습/추론용 후보 마스크 (불가 노드 제거)

    g = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=y, mask=mask,
        current_node=torch.tensor([current_node], dtype=torch.long),
        current_time=torch.tensor([current_time], dtype=torch.float32),
        visited=visited.float(),
    )
    return g

# ----------------- 간단한 어댑터: 인스턴스 로딩 -----------------

def load_instance_json(fp: Path) -> Tuple[Tensor, Tensor]:
    """
    매우 단순한 포맷(.json):
    {
      "coords": [[x,y], ...],
      "windows": [[l,u], ...]
    }
    -> PIP-constraint 데이터 로더를 이미 쓰고 있다면,
       그 로더가 반환하는 텐서로 coords, windows만 넘겨도 됨.
    """
    obj = json.loads(fp.read_text())
    coords = torch.tensor(obj["coords"], dtype=torch.float32)
    windows = torch.tensor(obj["windows"], dtype=torch.float32)
    return coords, windows

def generate_synthetic_instance(n: int = 50, seed: int = 0) -> Tuple[Tensor, Tensor]:
    random.seed(seed); torch.manual_seed(seed)
    coords = torch.rand(n, 2)
    l = torch.rand(n) * 0.7
    u = l + 0.2 + torch.rand(n) * 0.3  # 폭 0.2~0.5
    windows = torch.stack([l, u], dim=1)
    return coords, windows

# ----------------- 간단한 rollout → 학습 샘플 생성 -----------------

def greedy_states(coords: Tensor, windows: Tensor, start: int = 0) -> List[Dict]:
    """
    '가장 단순' 스텝 상태 생성기.
    - 현재 시각은 누적 이동거리로 정의.
    - 방문하지 않은 노드 중 u_i가 작은 순(EDF)으로 탐욕 방문.
    각 스텝에서 (cur, t, visited)로 그래프를 만들 수 있음.
    """
    N = coords.size(0)
    visited = torch.zeros(N, dtype=torch.bool)
    cur = start
    t = 0.0
    seq = []
    for _ in range(N - 1):
        seq.append({"current_node": int(cur), "current_time": float(t), "visited": visited.clone()})
        candidates = [i for i in range(N) if not visited[i] and i != cur]
        if not candidates:
            break
        # EDD(earliest due date)
        c_sorted = sorted(candidates, key=lambda i: float(windows[i, 1]))
        nxt = c_sorted[0]
        t = t + float(_euclid(coords[cur], coords[nxt]))
        visited[cur] = True
        cur = nxt
    # 마지막 스텝도 하나 더
    seq.append({"current_node": int(cur), "current_time": float(t), "visited": visited.clone()})
    return seq

def build_graph_batch_from_instance(coords: Tensor, windows: Tensor) -> List[Data]:
    states = greedy_states(coords, windows, start=0)
    graphs = []
    for s in states:
        g = build_step_graph(coords, windows,
                             s["current_node"], s["current_time"], s["visited"])
        graphs.append(g)
    return graphs
