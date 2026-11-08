"""Auth model: predicts from graph structure whether most of an edge's events use NTLM.

Same encoder as main_model.py with an MLP head on [z_u * z_v, |z_u - z_v|].

python auth_model.py DATA_DIR OUT_DIR --pairs PAIRS.npz [--seed 0]
"""
import argparse, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from data import LanlWindows, score_pairs
from main_model import Encoder


class AuthModel(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        self.enc = Encoder(n_nodes)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def edge_logit(self, z, ei):
        zu, zv = z[ei[0]], z[ei[1]]
        return self.head(torch.cat([zu * zv, (zu - zv).abs()], -1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--max_day', type=int, default=None)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--holdout', type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(a.seed); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out, exist_ok=True)
    D = LanlWindows(a.data, max_day=a.max_day)
    rng = np.random.default_rng(a.seed)
    seq = []
    for w in D.train_windows[:-a.holdout]:
        e = D.window_edges(w)
        ei = D.edge_index(w, dev)
        y = torch.from_numpy((e['ntlm'] / np.maximum(e['cnt'], 1) > 0.5).astype(np.float32)).to(dev)
        m = torch.from_numpy(rng.random(len(y)) < 0.05).to(dev)
        seq.append((ei, y, m))
    model = AuthModel(D.n_nodes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    best, bad, t0 = -1, 0, time.time()
    for ep in range(a.epochs):
        model.train(); opt.zero_grad(); h = None; loss = 0
        for ei, y, m in seq:
            h = model.enc.gru(model.enc.snapshot(ei[:, ~m]), h)
            loss = loss + F.binary_cross_entropy_with_logits(model.edge_logit(h, ei[:, ~m]), y[~m])
        (loss / len(seq)).backward(); opt.step()
        model.eval(); h = None; ps, ys = [], []
        with torch.no_grad():
            for ei, y, m in seq:
                h = model.enc.gru(model.enc.snapshot(ei[:, ~m]), h)
                if m.any():
                    ps.append(model.edge_logit(h, ei[:, m]).cpu()); ys.append(y[m].cpu())
        auc = roc_auc_score(torch.cat(ys).numpy(), torch.cat(ps).numpy())
        print(f'ep {ep} loss {loss.item()/len(seq):.4f} val_auc {auc:.4f} {time.time()-t0:.0f}s', flush=True)
        if auc > best:
            best, bad = auc, 0; torch.save(model.state_dict(), os.path.join(a.out, f'auth_seed{a.seed}.pt'))
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(torch.load(os.path.join(a.out, f'auth_seed{a.seed}.pt'))); model.eval()
    pairs = dict(np.load(a.pairs))
    out = score_pairs(D, pairs, lambda ei, h: model.enc.gru(model.enc.snapshot(ei), h), model.edge_logit, dev)
    np.savez_compressed(os.path.join(a.out, f'auth_scores_seed{a.seed}.npz'), logit=out, best_val_auc=best)
    print('done', best)


if __name__ == '__main__':
    main()
