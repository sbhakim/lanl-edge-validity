"""EncG: SEAL link prediction (Zhang and Chen, NeurIPS 2018) on 2-hop enclosing subgraphs.

The subgraph of (u, v) is taken from the window's undirected graph with the target link
removed. Nodes get double-radius labels and the model is a 3-layer GCN with mean/max
pooling. Positives are observed training edges, negatives sampled non-edges.
Hub nodes have thousands of neighbours, so neighbourhoods are sampled (MAX_NB per hop)
and subgraphs are capped at 200 nodes.

python encg.py DATA_DIR OUT_DIR --pairs PAIRS.npz [--seed 0] [--workers 48]
"""
import argparse, os, time
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from data import LanlWindows

K_CAP = 3


def build_adj(src, dst, n):
    a = sp.coo_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.int8).tocsr()
    a.setdiag(0); a.eliminate_zeros()
    return a


MAX_NB = 50


def nbrs(adj, u, rng, exclude=None):
    nb = adj.indices[adj.indptr[u]:adj.indptr[u + 1]]
    if exclude is not None:
        nb = nb[nb != exclude]
    if len(nb) > MAX_NB:
        nb = rng.choice(nb, MAX_NB, replace=False)
    return nb


def hop_dist(adj, root, other, rng):
    """Hop distance (0-2) from root, not passing through the target link."""
    n1 = nbrs(adj, root, rng, exclude=other)
    n2 = np.unique(np.concatenate([nbrs(adj, x, rng) for x in n1])) if len(n1) else np.empty(0, np.int64)
    d = dict.fromkeys(n2.tolist(), 2)
    d.update(dict.fromkeys(n1.tolist(), 1))
    d[root] = 0
    return d


def subgraph(adj, u, v, max_nodes=200, rng=None):
    rng = rng or np.random.default_rng(0)
    du, dv = hop_dist(adj, u, v, rng), hop_dist(adj, v, u, rng)
    rest = [n for n in dict.fromkeys(list(du) + list(dv)) if n not in (u, v)]
    if len(rest) > max_nodes - 2:
        close = [n for n in rest if du.get(n, K_CAP) <= 1 or dv.get(n, K_CAP) <= 1]
        far = [n for n in rest if not (du.get(n, K_CAP) <= 1 or dv.get(n, K_CAP) <= 1)]
        rest = close[:max_nodes - 2]
        if len(rest) < max_nodes - 2 and far:
            rest += list(rng.choice(far, size=min(len(far), max_nodes - 2 - len(rest)), replace=False))
    nodes = [u, v] + rest
    sub = adj[nodes][:, nodes].tocoo()
    m = ~(((sub.row == 0) & (sub.col == 1)) | ((sub.row == 1) & (sub.col == 0)))  # drop target edge
    ei = np.stack([sub.row[m], sub.col[m]]).astype(np.int64)
    lab = np.array([min(du.get(n, K_CAP), K_CAP) * (K_CAP + 1) + min(dv.get(n, K_CAP), K_CAP) for n in nodes])
    return ei, lab


def _work(args):
    w, src, dst, us, vs = args
    n = int(max(src.max(initial=0), dst.max(initial=0), us.max(initial=0), vs.max(initial=0))) + 1
    adj = build_adj(src, dst, n)
    rng = np.random.default_rng(int(w))
    return [subgraph(adj, int(u), int(v), rng=rng) for u, v in zip(us, vs)]


def to_batch(items):
    return Batch.from_data_list([Data(edge_index=torch.from_numpy(ei), x=torch.from_numpy(lab).long(),
                                      num_nodes=len(lab)) for ei, lab in items])


class SEAL(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.emb = nn.Embedding((K_CAP + 1) ** 2, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(3)])
        self.mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, b):
        x = self.emb(b.x)
        for c in self.convs:
            x = torch.relu(c(x, b.edge_index))
        return self.mlp(torch.cat([global_mean_pool(x, b.batch), global_max_pool(x, b.batch)], -1)).squeeze(-1)


def extract(D, wins, pairs_per_win, workers):
    jobs = []
    for w in wins:
        e = D.window_edges(w)
        us, vs = pairs_per_win[w]
        jobs.append((w, e['src'], e['dst'], us, vs))
    with Pool(workers) as p:
        return p.map(_work, jobs, chunksize=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('out')
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--holdout', type=int, default=4)
    ap.add_argument('--train_per_win', type=int, default=2000)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--max_day', type=int, default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out, exist_ok=True)
    D = LanlWindows(a.data, max_day=a.max_day)
    t0 = time.time()

    tw = D.train_windows[:-a.holdout]
    req, ylab = {}, []
    for w in tw:
        e = D.window_edges(w)
        k = min(a.train_per_win, len(e['src']))
        pi = rng.choice(len(e['src']), k, replace=False)
        nodes = np.unique(np.r_[e['src'], e['dst']])
        ex = set(zip(e['src'].tolist(), e['dst'].tolist()))
        nu, nv = [], []
        while len(nu) < k:
            u, v = rng.choice(nodes), rng.choice(nodes)
            if u != v and (u, v) not in ex:
                nu.append(u); nv.append(v)
        req[w] = (np.r_[e['src'][pi], nu], np.r_[e['dst'][pi], nv])
        ylab.append(np.r_[np.ones(k), np.zeros(k)])
    items = [s for lst in extract(D, tw, req, a.workers) for s in lst]
    y = torch.from_numpy(np.concatenate(ylab)).float()
    print(f'train subgraphs {len(items):,} {time.time()-t0:.0f}s', flush=True)
    perm = rng.permutation(len(items)); nval = len(items) // 20
    va, tr = perm[:nval], perm[nval:]

    model = SEAL().to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = -1
    for ep in range(a.epochs):
        model.train()
        for i in range(0, len(tr), 512):
            ix = tr[i:i + 512]
            b = to_batch([items[j] for j in ix]).to(dev)
            loss = F.binary_cross_entropy_with_logits(model(b), y[ix].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); ps = []
        with torch.no_grad():
            for i in range(0, len(va), 2048):
                ps.append(model(to_batch([items[j] for j in va[i:i + 2048]]).to(dev)).cpu())
        auc = roc_auc_score(y[va].numpy(), torch.cat(ps).numpy())
        print(f'ep {ep} val_auc {auc:.4f} {time.time()-t0:.0f}s', flush=True)
        if auc > best:
            best = auc; torch.save(model.state_dict(), os.path.join(a.out, f'encg_seed{a.seed}.pt'))

    model.load_state_dict(torch.load(os.path.join(a.out, f'encg_seed{a.seed}.pt'))); model.eval()
    P = dict(np.load(a.pairs))
    out = np.zeros(len(P['win']), np.float32)
    wins = np.unique(P['win']).tolist()
    for s in range(0, len(wins), 64):                 # blocks of windows to bound memory
        blk = wins[s:s + 64]
        req, ids = {}, {}
        for w in blk:
            ix = np.flatnonzero(P['win'] == w); ids[w] = ix
            req[w] = (P['src'][ix], P['dst'][ix])
        res = extract(D, blk, req, a.workers)
        with torch.no_grad():
            for w, sg in zip(blk, res):
                vals = []
                for i in range(0, len(sg), 4096):
                    vals.append(model(to_batch(sg[i:i + 4096]).to(dev)).cpu())
                out[ids[w]] = torch.cat(vals).numpy() if vals else []
        print(f'scored windows {s + len(blk)}/{len(wins)} {time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(os.path.join(a.out, f'encg_scores_seed{a.seed}.npz'), logit=out, best_val_auc=best)
    print('done', best)


if __name__ == '__main__':
    main()
