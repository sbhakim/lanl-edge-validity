"""Pairs scored by every detector.

fit:  the last --holdout pre-attack windows, which no model is trained on. Observed edges
      are labelled 0 and an equal number of sampled non-edges 1 (proxy anomalies).
test: every malicious edge from the first red-team window on, plus a uniform sample of
      --n_benign benign edges.

python make_pairs.py DATA_DIR OUT.npz [--holdout 4] [--n_benign 2000000] [--seed 0]
"""
import argparse
import numpy as np
from data import LanlWindows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('out')
    ap.add_argument('--holdout', type=int, default=4)
    ap.add_argument('--n_benign', type=int, default=2_000_000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_day', type=int, default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    D = LanlWindows(a.data, max_day=a.max_day)
    fit_w = D.train_windows[-a.holdout:]
    W, S, T, Y, SPLIT = [], [], [], [], []
    for w in fit_w:
        e = D.window_edges(w)
        n = len(e['src'])
        existing = set(zip(e['src'].tolist(), e['dst'].tolist()))
        nodes = np.unique(np.r_[e['src'], e['dst']])
        ns, nd = [], []
        while len(ns) < n:
            u, v = rng.choice(nodes), rng.choice(nodes)
            if u != v and (u, v) not in existing:
                ns.append(u); nd.append(v)
        W += [np.full(2 * n, w)]; S += [e['src'], np.array(ns)]; T += [e['dst'], np.array(nd)]
        Y += [np.zeros(n), np.ones(n)]; SPLIT += [np.zeros(2 * n)]
    te = np.isin(D.c['win'], D.test_windows)
    idx = np.flatnonzero(te)
    mal = idx[D.c['label'][idx] == 1]
    ben = idx[D.c['label'][idx] == 0]
    ben = rng.choice(ben, size=min(a.n_benign, len(ben)), replace=False)
    sel = np.sort(np.r_[mal, ben])
    W.append(D.c['win'][sel]); S.append(D.c['src'][sel]); T.append(D.c['dst'][sel])
    Y.append(D.c['label'][sel]); SPLIT.append(np.ones(len(sel)))
    cat = lambda L, t: np.concatenate(L).astype(t)
    out = dict(win=cat(W, np.int32), src=cat(S, np.int32), dst=cat(T, np.int32), y=cat(Y, np.int8),
               split=cat(SPLIT, np.int8), holdout_windows=np.array(fit_w),
               n_test_benign_total=np.int64(te.sum() - len(mal)))
    np.savez_compressed(a.out, **out)
    print(f'fit pairs {int((out["split"]==0).sum()):,}  test pairs {int((out["split"]==1).sum()):,} '
          f'(malicious {len(mal)}, benign sampled {len(ben):,} of {int(out["n_test_benign_total"]):,})')


if __name__ == '__main__':
    main()
