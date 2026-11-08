"""Prospective scoring of the main model (protocols/round1.txt, G).

G1: the trained detector, but window t is scored with the state after window t-1.
G2: the same model trained to predict window t from the state after t-1; everything
    else as in src/main_model.py.

python prospective.py ROOT OUT_DIR --seed S
"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('out'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--patience', type=int, default=5)
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from torch_geometric.utils import negative_sampling
from data import LanlWindows
from main_model import Encoder, logits

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = 'cuda'
os.makedirs(a.out, exist_ok=True)
D = LanlWindows(os.path.join(a.root, 'data', 'prep'))
P = dict(np.load(os.path.join(a.root, 'results', 'pairs.npz')))


def score_prospective(model):
    """Score window w with the state after window w-1."""
    out = np.full(len(P['win']), np.nan, np.float32)
    order = np.argsort(P['win'], kind='stable'); pw = P['win'][order]
    h = None
    with torch.no_grad():
        for w in D.wins.tolist():
            lo, hi = np.searchsorted(pw, w, 'left'), np.searchsorted(pw, w, 'right')
            if hi > lo and h is not None:
                ix = order[lo:hi]
                pei = torch.from_numpy(np.stack([P['src'][ix], P['dst'][ix]]).astype(np.int64)).to(dev)
                out[ix] = logits(h, pei).cpu().numpy()
            h = model.gru(model.snapshot(D.edge_index(w, dev)), h)
    return out


# G1
m1 = Encoder(D.n_nodes).to(dev)
m1.load_state_dict(torch.load(os.path.join(a.root, 'results', f'main_seed{a.seed}.pt'))); m1.eval()
g1 = score_prospective(m1)

# G2
tw = D.train_windows[:-4]
rng = np.random.default_rng(a.seed)
tr_ei, va_ei = [], []
for w in tw:
    ei = D.edge_index(w); msk = rng.random(ei.shape[1]) < 0.05
    tr_ei.append(ei[:, ~torch.from_numpy(msk)].to(dev)); va_ei.append(ei[:, torch.from_numpy(msk)].to(dev))
m2 = Encoder(D.n_nodes).to(dev)
opt = torch.optim.Adam(m2.parameters(), lr=0.005)
best, bad, t0 = -1, 0, time.time()
ck = os.path.join(a.out, f'pred_seed{a.seed}.pt')
for ep in range(a.epochs):
    m2.train(); opt.zero_grad()
    zs, _ = m2.forward_seq(tr_ei)
    loss = 0
    for z, ei in zip(zs[:-1], tr_ei[1:]):          # state after t-1 predicts window t
        neg = negative_sampling(ei, num_nodes=D.n_nodes, num_neg_samples=ei.shape[1])
        loss = loss + F.binary_cross_entropy_with_logits(logits(z, ei), torch.ones(ei.shape[1], device=dev)) \
                    + F.binary_cross_entropy_with_logits(logits(z, neg), torch.zeros(neg.shape[1], device=dev))
    (loss / (len(zs) - 1)).backward(); opt.step()
    m2.eval()
    with torch.no_grad():
        zs, _ = m2.forward_seq(tr_ei)
        ys, ps = [], []
        for z, ei, ve in zip(zs[:-1], tr_ei[1:], va_ei[1:]):
            if ve.shape[1] == 0: continue
            neg = negative_sampling(ei, num_nodes=D.n_nodes, num_neg_samples=ve.shape[1])
            ps += [logits(z, ve).cpu(), logits(z, neg).cpu()]; ys += [torch.ones(ve.shape[1]), torch.zeros(neg.shape[1])]
        auc = roc_auc_score(torch.cat(ys).numpy(), torch.cat(ps).numpy())
    print(f'ep {ep} val_auc {auc:.4f} {time.time()-t0:.0f}s', flush=True)
    if auc > best:
        best, bad = auc, 0; torch.save(m2.state_dict(), ck)
    else:
        bad += 1
        if bad >= a.patience: break
m2.load_state_dict(torch.load(ck)); m2.eval()
g2 = score_prospective(m2)
np.savez_compressed(os.path.join(a.out, f'prosp_scores_seed{a.seed}.npz'), g1=g1, g2=g2, g2_best_val_auc=best)

# evaluation on the test pairs; anomaly score = -logit
te = P['split'] == 1
names = np.load(os.path.join(a.root, 'data', 'prep', 'nodes.npy'), allow_pickle=True)
y = P['y'][te].astype(int); srcn = names[P['src'][te]]; keep = srcn != 'C17693'
retro = -np.load(os.path.join(a.root, 'results', f'main_scores_seed{a.seed}.npz'))['logit'][te]


def tp_at(yy, s, f=0.005):
    fp, tp, _ = roc_curve(yy, s); return float(np.interp(f, fp, tp))


res = {'g2_best_val_auc': float(best)}
for k, s in {'retrospective (paper)': retro, 'G1 frozen, state t-1': -g1[te], 'G2 trained predictor': -g2[te]}.items():
    assert np.isfinite(s).all(), k
    thr = np.quantile(s[y == 0], 0.995)
    res[k] = {'auc_all': float(roc_auc_score(y, s)), 'tp_all': tp_at(y, s), 'auc_excl': float(roc_auc_score(y[keep], s[keep])),
              'tp_excl': tp_at(y[keep], s[keep]),
              'hosts_k': {h: int(((srcn == h) & (y == 1) & (s > thr)).sum()) for h in sorted(set(srcn[y == 1]))}}
json.dump(res, open(os.path.join(a.out, f'prosp_eval_seed{a.seed}.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
