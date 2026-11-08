"""Per-window edge lists produced by prep_lanl.py."""
import glob, json, os
import numpy as np
import torch

DATE_OF_EVIL = 150885  # first red-team event, in seconds (same constant as Euler)


class LanlWindows:
    def __init__(self, root, window=1800, max_day=None):
        self.meta = json.load(open(os.path.join(root, 'meta.json')))
        self.n_nodes = self.meta['nodes']
        self.window = window
        files = sorted(glob.glob(os.path.join(root, 'day_*.npz')))
        if max_day is not None:
            files = files[:max_day + 1]
        cols = {k: [] for k in ['win', 'src', 'dst', 'cnt', 'ntlm', 'label', 'nov_dst', 'nov_src']}
        for f in files:
            z = np.load(f)
            for k in cols:
                cols[k].append(z[k])
        c = {k: np.concatenate(v) for k, v in cols.items()}
        # computer-to-computer edges only, no self-loops
        names = np.load(os.path.join(root, 'nodes.npy'), allow_pickle=True)
        isC = np.array([str(n).startswith('C') for n in names])
        keep = (c['src'] != c['dst']) & isC[c['src']] & isC[c['dst']]
        c = {k: v[keep] for k, v in c.items()}
        order = np.argsort(c['win'], kind='stable')
        c = {k: v[order] for k, v in c.items()}
        self.c = c
        wins, starts = np.unique(c['win'], return_index=True)
        self.wins = wins
        self.bounds = dict(zip(wins.tolist(), zip(starts.tolist(), np.r_[starts[1:], len(c['win'])].tolist())))
        self.evil_win = DATE_OF_EVIL // window

    def window_edges(self, w):
        """Edge arrays of window w (empty if the window has no edges)."""
        if w not in self.bounds:
            return {k: v[:0] for k, v in self.c.items()}
        s, e = self.bounds[w]
        return {k: v[s:e] for k, v in self.c.items()}

    def edge_index(self, w, device='cpu'):
        e = self.window_edges(w)
        return torch.from_numpy(np.stack([e['src'], e['dst']]).astype(np.int64)).to(device)

    @property
    def train_windows(self):
        return [w for w in self.wins.tolist() if w < self.evil_win]

    @property
    def test_windows(self):
        return [w for w in self.wins.tolist() if w >= self.evil_win]


def score_pairs(D, pairs, step_fn, edge_fn, dev):
    """Run the encoder over every window in order and score the pairs of each window
    with the state after that window. step_fn(ei, h) -> h, edge_fn(h, pairs) -> logits."""
    out = np.zeros(len(pairs['win']), dtype=np.float32)
    order = np.argsort(pairs['win'], kind='stable')
    pw = pairs['win'][order]
    h = None
    with torch.no_grad():
        for w in D.wins.tolist():
            h = step_fn(D.edge_index(w, dev), h)
            lo, hi = np.searchsorted(pw, w, 'left'), np.searchsorted(pw, w, 'right')
            if hi > lo:
                ix = order[lo:hi]
                pei = torch.from_numpy(np.stack([pairs['src'][ix], pairs['dst'][ix]]).astype(np.int64)).to(dev)
                out[ix] = edge_fn(h, pei).cpu().numpy()
    return out
