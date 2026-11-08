"""EncG burst intervention (protocols/round2.txt, R2-3).

Each malicious test edge is scored by the trained EncG twice: on its window's graph, and
on the same graph without the window's other malicious edges. Both runs use the same
neighbour-sampling seed. A removed edge can survive as an undirected link if a benign
edge runs the other way in the same window.

python encg_burst.py ROOT OUT_JSON
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from data import LanlWindows
from encg import SEAL, build_adj, subgraph, to_batch

ROOT, OUTF = sys.argv[1:3]
R = os.path.join(ROOT, 'results')
P = dict(np.load(os.path.join(R, 'pairs.npz')))
names = np.load(os.path.join(ROOT, 'data', 'prep', 'nodes.npy'), allow_pickle=True)
te = P['split'] == 1
mi = np.flatnonzero(te & (P['y'] == 1))
D = LanlWindows(os.path.join(ROOT, 'data', 'prep'))
dev = 'cuda'

items_c, items_i, meta = [], [], []
for w in np.unique(P['win'][mi]).tolist():
    e = D.window_edges(w)
    lab = e['label'] > 0
    n = int(max(e['src'].max(), e['dst'].max())) + 1
    adj_c = build_adj(e['src'], e['dst'], n)
    adj_i = build_adj(e['src'][~lab], e['dst'][~lab], n)
    idx = mi[P['win'][mi] == w]
    for j in idx:
        u, v = int(P['src'][j]), int(P['dst'][j])
        sd = (w * 1_000_003 + u * 7919 + v) % (2 ** 32)
        items_c.append(subgraph(adj_c, u, v, rng=np.random.default_rng(sd)))
        items_i.append(subgraph(adj_i, u, v, rng=np.random.default_rng(sd)))
        meta.append((int(j), w, str(names[u]), int(lab.sum()) - 1))
print('subgraphs', len(meta), flush=True)

out = {'n_edges': len(meta), 'removed_per_target': {}, 'seeds': {}}
rem = np.array([m[3] for m in meta]); hostv = np.array([m[2] for m in meta]); jj = np.array([m[0] for m in meta])
for sd in range(5):
    model = SEAL().to(dev); model.load_state_dict(torch.load(os.path.join(R, f'encg_seed{sd}.pt'))); model.eval()

    def score(items):
        with torch.no_grad():
            return np.concatenate([model(to_batch(items[i:i + 2048]).to(dev)).cpu().numpy() for i in range(0, len(items), 2048)])
    sc, si = -score(items_c), -score(items_i)
    cached = -np.load(os.path.join(R, f'encg_scores_seed{sd}.npz'))['logit']
    thr = np.quantile(cached[te & (P['y'] == 0)], .995)
    o = {'corr_control_vs_cached': float(np.corrcoef(sc, cached[jj])[0, 1]),
         'hits_cached': int((cached[jj] > thr).sum()), 'hits_control': int((sc > thr).sum()), 'hits_intervention': int((si > thr).sum()),
         'by_host': {}}
    for h in ['C17693', 'C19932', 'C22409', 'C18025']:
        m = hostv == h
        o['by_host'][h] = {'n': int(m.sum()), 'mean_removed': float(rem[m].mean()),
                           'mean_delta_anomaly': float((si[m] - sc[m]).mean()),
                           'hits_control': int((sc[m] > thr).sum()), 'hits_intervention': int((si[m] > thr).sum()),
                           'lost': int(((sc[m] > thr) & ~(si[m] > thr)).sum()), 'gained': int((~(sc[m] > thr) & (si[m] > thr)).sum())}
    bins = {'0': rem == 0, '1-4': (rem >= 1) & (rem <= 4), '5-19': (rem >= 5) & (rem <= 19), '>=20': rem >= 20}
    o['by_removed'] = {k: {'n': int(m.sum()), 'mean_delta_anomaly': float((si[m] - sc[m]).mean()) if m.any() else None,
                           'hits_control': int((sc[m] > thr).sum()), 'hits_intervention': int((si[m] > thr).sum())}
                       for k, m in bins.items()}
    out['seeds'][sd] = o
    print(sd, json.dumps({k: v for k, v in o.items() if k != 'by_removed'}), flush=True)
out['removed_per_target'] = {h: [int(x) for x in np.percentile(rem[hostv == h], [0, 50, 100])] for h in set(hostv)}
json.dump(out, open(OUTF, 'w'), indent=1)
