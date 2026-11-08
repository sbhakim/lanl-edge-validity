"""Main model: Euler's detector, a GCN per window followed by a GRU.

Settings follow Euler (King and Huang, NDSS 2023): two GCN layers (hidden 32, embedding 16),
lr 0.005, early stopping with patience 5, negative sampling, and an inner-product decoder.
5% of each training window's edges are held out for validation.

python main_model.py DATA_DIR OUT_DIR --pairs PAIRS.npz [--seed 0]
Writes main_scores_seed{S}.npz with the edge logit of every pair, in pair order.
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from data import LanlWindows, score_pairs


class Encoder(nn.Module):
    def __init__(self, n_nodes, hidden=32, z=16):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, hidden)       # one-hot node features
        self.c1 = GCNConv(hidden, hidden, add_self_loops=True)
        self.c2 = GCNConv(hidden, hidden, add_self_loops=True)
        self.gru = nn.GRUCell(hidden, z)
        self.drop = nn.Dropout(0.25)

    def snapshot(self, ei):
        x = self.emb.weight
        x = self.drop(torch.tanh(self.c1(x, ei)))
        return torch.tanh(self.c2(x, ei))

    def forward_seq(self, eis, h=None):
        zs = []
        for ei in eis:
            h = self.gru(self.snapshot(ei), h)
            zs.append(h)
        return zs, h


def logits(z, ei):
    return (z[ei[0]] * z[ei[1]]).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--max_day', type=int, default=None)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--holdout', type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out, exist_ok=True)
    D = LanlWindows(a.data, max_day=a.max_day)
    tw = D.train_windows[:-a.holdout]   # the last windows are kept for fitting rule weights
    rng = np.random.default_rng(a.seed)
    tr_ei, va_ei = [], []
    for w in tw:
        ei = D.edge_index(w)
        m = rng.random(ei.shape[1]) < 0.05
        tr_ei.append(ei[:, ~torch.from_numpy(m)].to(dev)); va_ei.append(ei[:, torch.from_numpy(m)].to(dev))

    model = Encoder(D.n_nodes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, bad, t0 = -1, 0, time.time()
    for ep in range(a.epochs):
        model.train(); opt.zero_grad()
        zs, _ = model.forward_seq(tr_ei)
        loss = 0
        for z, ei in zip(zs, tr_ei):
            neg = negative_sampling(ei, num_nodes=D.n_nodes, num_neg_samples=ei.shape[1])
            loss = loss + F.binary_cross_entropy_with_logits(logits(z, ei), torch.ones(ei.shape[1], device=dev)) \
                        + F.binary_cross_entropy_with_logits(logits(z, neg), torch.zeros(neg.shape[1], device=dev))
        (loss / len(zs)).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            zs, _ = model.forward_seq(tr_ei)
            ys, ps = [], []
            for z, ei, ve in zip(zs, tr_ei, va_ei):
                if ve.shape[1] == 0: continue
                neg = negative_sampling(ei, num_nodes=D.n_nodes, num_neg_samples=ve.shape[1])
                ps += [logits(z, ve).cpu(), logits(z, neg).cpu()]
                ys += [torch.ones(ve.shape[1]), torch.zeros(neg.shape[1])]
            auc = roc_auc_score(torch.cat(ys).numpy(), torch.cat(ps).numpy())
        print(f'ep {ep} loss {loss.item()/len(zs):.4f} val_auc {auc:.4f} {time.time()-t0:.0f}s', flush=True)
        if auc > best:
            best, bad = auc, 0
            torch.save(model.state_dict(), os.path.join(a.out, f'main_seed{a.seed}.pt'))
        else:
            bad += 1
            if bad >= a.patience: break

    model.load_state_dict(torch.load(os.path.join(a.out, f'main_seed{a.seed}.pt')))
    model.eval()
    pairs = dict(np.load(a.pairs))
    out = score_pairs(D, pairs, lambda ei, h: model.gru(model.snapshot(ei), h), logits, dev)
    np.savez_compressed(os.path.join(a.out, f'main_scores_seed{a.seed}.npz'), logit=out, best_val_auc=best)
    print('done', best)


if __name__ == '__main__':
    main()
