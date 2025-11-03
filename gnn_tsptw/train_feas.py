# cvrptw/POMO+PIP/gnn_tsptw/train_feas.py
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from .dataset_tsptw import (
    load_instance_json, generate_synthetic_instance,
    build_graph_batch_from_instance
)
from .model_gnn_tsptw import TSPTWFeasGIN

def make_step_graphs_from_dir(data_dir: Path, limit: int = 50):
    files = list(data_dir.glob("*.json"))
    graphs = []
    for fp in files[:limit]:
        coords, windows = load_instance_json(fp)
        graphs.extend(build_graph_batch_from_instance(coords, windows))
    return graphs

def make_step_graphs_synth(n_inst=32, n_nodes=50, seed=0):
    graphs = []
    for i in range(n_inst):
        coords, windows = generate_synthetic_instance(n_nodes, seed + i)
        graphs.extend(build_graph_batch_from_instance(coords, windows))
    return graphs

def train_epoch(model, graphs, optim, device):
    model.train()
    total = 0.0
    for g in graphs:
        g = g.to(device)
        logits = model(g)             # (N,)
        mask = g.mask > 0.5           # 후보만 학습
        if mask.sum() == 0:           # 전부 불가면 스킵
            continue
        loss = F.binary_cross_entropy_with_logits(logits[mask], g.y[mask])
        optim.zero_grad(); loss.backward(); optim.step()
        total += float(loss.item())
    return total / max(1, len(graphs))

@torch.no_grad()
def eval_epoch(model, graphs, device):
    model.eval()
    tot_f1, tot = 0.0, 0
    for g in graphs:
        g = g.to(device)
        logits = model(g)
        mask = g.mask > 0.5
        if mask.sum() == 0:
            continue
        pred = (torch.sigmoid(logits[mask]) > 0.5).float()
        y = g.y[mask]
        tp = (pred.eq(1) & y.eq(1)).sum().item()
        fp = (pred.eq(1) & y.eq(0)).sum().item()
        fn = (pred.eq(0) & y.eq(1)).sum().item()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        tot_f1 += f1; tot += 1
    return tot_f1 / max(1, tot)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.data_dir:
        graphs = make_step_graphs_from_dir(Path(args.data_dir))
    else:
        graphs = make_step_graphs_synth(n_inst=32, n_nodes=50, seed=0)

    # 간단 분할
    split = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:split], graphs[split:]

    model = TSPTWFeasGIN(in_dim=4, hid=128, layers=3).to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_graphs, optim, device)
        val_f1  = eval_epoch(model, val_graphs, device)
        print(f"[ep {ep:02d}] loss={tr_loss:.4f}  val_f1={val_f1:.4f}")

if __name__ == "__main__":
    main()
