"""Deduplicate LANL authentication events into 30-minute directed edges.

All authentication types are kept. Euler's loader keeps NTLM only, which would make
the rule [main=Mal] => [auth=NTLM] true by construction.

Writes to OUT_DIR:
  day_XXX.npz  win, src, dst, cnt, ntlm, label, nov_dst, nov_src (one file per day)
               nov_dst: events whose (user, destination) pair is new, i.e. first access
               nov_src: events whose (user, source) pair is new
  nodes.npy    computer name of each node id
  meta.json    counts

python prep_lanl.py AUTH_GZ REDTEAM_GZ OUT_DIR [--window 1800]
"""
import argparse, gzip, json, os, time
import numpy as np
import pandas as pd

COLS = ['ts', 'src_user', 'dst_user', 'src', 'dst', 'auth', 'logon', 'orient', 'ok']


def has_header(path):
    with gzip.open(path, 'rt') as f:
        first = f.readline().split(',')[0]
    return not first.strip().isdigit()


def load_redteam(path):
    rt = pd.read_csv(path, header=None, names=['ts', 'user', 'src', 'dst'],
                     skiprows=1 if has_header(path) else 0)
    return rt[['ts', 'src', 'dst']].drop_duplicates()


class NodeMap:
    def __init__(self):
        self.d = {}

    def map(self, s: pd.Series) -> np.ndarray:
        new = pd.unique(s[~s.isin(self.d.keys())])
        for n in new:
            self.d[n] = len(self.d)
        return s.map(self.d).to_numpy(np.int32)

    def names(self):
        out = [None] * len(self.d)
        for k, v in self.d.items():
            out[v] = k
        return np.array(out, dtype=object)


def aggregate(df):
    g = df.groupby(['win', 'src', 'dst'], sort=False)
    return g.agg(cnt=('ntlm', 'size'), ntlm=('ntlm', 'sum'), label=('label', 'max'),
                 nov_dst=('nov_dst', 'sum'), nov_src=('nov_src', 'sum')).reset_index()


class FirstSeen:
    """Flags the first occurrence of each int64 key; chunks must arrive in time order."""
    def __init__(self):
        self.seen = np.empty(0, np.int64)

    def flags(self, keys):
        uniq, first = np.unique(keys, return_index=True)
        new = ~np.isin(uniq, self.seen, assume_unique=True)
        out = np.zeros(len(keys), np.int32)
        out[first[new]] = 1
        self.seen = np.union1d(self.seen, uniq[new])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('auth'); ap.add_argument('redteam'); ap.add_argument('out')
    ap.add_argument('--window', type=int, default=1800)
    ap.add_argument('--chunksize', type=int, default=20_000_000)
    ap.add_argument('--max_rows', type=int, default=0, help='stop after N rows, for quick tests')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rt = load_redteam(a.redteam)
    rt['label'] = np.int8(1)
    nm, um = NodeMap(), NodeMap()
    fs_dst, fs_src = FirstSeen(), FirstSeen()
    per_day = 86400 // a.window
    pending = []          # aggregated rows of the current day
    carry = None          # last window of a chunk may continue in the next chunk
    cur_day, n_rows, n_edges, n_mal = 0, 0, 0, 0
    t0 = time.time()

    def flush(day, frames):
        nonlocal n_edges, n_mal
        if not frames:
            return
        agg = pd.concat(frames)
        agg = agg.groupby(['win', 'src', 'dst'], sort=True).agg(
            cnt=('cnt', 'sum'), ntlm=('ntlm', 'sum'), label=('label', 'max'),
            nov_dst=('nov_dst', 'sum'), nov_src=('nov_src', 'sum')).reset_index()
        np.savez_compressed(os.path.join(a.out, f'day_{day:03d}.npz'),
                            win=agg.win.to_numpy(np.int32), src=agg.src.to_numpy(np.int32),
                            dst=agg.dst.to_numpy(np.int32), cnt=agg.cnt.to_numpy(np.int32),
                            ntlm=agg.ntlm.to_numpy(np.int32), label=agg.label.to_numpy(np.int8),
                            nov_dst=agg.nov_dst.to_numpy(np.int32), nov_src=agg.nov_src.to_numpy(np.int32))
        n_edges += len(agg); n_mal += int(agg.label.sum())

    reader = pd.read_csv(a.auth, header=None, names=COLS, usecols=['ts', 'src_user', 'src', 'dst', 'auth'],
                         skiprows=1 if has_header(a.auth) else 0, chunksize=a.chunksize,
                         dtype={'ts': np.int64, 'src_user': str, 'src': str, 'dst': str, 'auth': str})
    for chunk in reader:
        n_rows += len(chunk)
        chunk = chunk.merge(rt, on=['ts', 'src', 'dst'], how='left')
        chunk['label'] = chunk['label'].fillna(0).astype(np.int8)
        chunk['ntlm'] = chunk['auth'].fillna('').str.upper().str.contains('NTLM').astype(np.int32)
        chunk['win'] = (chunk['ts'] // a.window).astype(np.int32)
        chunk['src'] = nm.map(chunk['src'].str.replace('$', '', regex=False))
        chunk['dst'] = nm.map(chunk['dst'].str.replace('$', '', regex=False))
        uid = um.map(chunk['src_user'].fillna('?')).astype(np.int64)
        chunk['nov_dst'] = fs_dst.flags((uid << 24) | chunk['dst'].to_numpy(np.int64))
        chunk['nov_src'] = fs_src.flags((uid << 24) | chunk['src'].to_numpy(np.int64))
        chunk = chunk[['win', 'src', 'dst', 'ntlm', 'label', 'nov_dst', 'nov_src']]
        if carry is not None:
            chunk = pd.concat([carry, chunk])
        last_win = chunk['win'].iloc[-1]
        carry = chunk[chunk['win'] == last_win]
        done = aggregate(chunk[chunk['win'] < last_win])
        for day, part in done.groupby(done['win'] // per_day):
            while day > cur_day:
                flush(cur_day, pending); pending = []; cur_day += 1
            pending.append(part)
        print(f'rows={n_rows:,} days<={cur_day} nodes={len(nm.d):,} {time.time()-t0:.0f}s', flush=True)
        if a.max_rows and n_rows >= a.max_rows:
            break
    if carry is not None and len(carry):
        done = aggregate(carry)
        for day, part in done.groupby(done['win'] // per_day):
            while day > cur_day:
                flush(cur_day, pending); pending = []; cur_day += 1
            pending.append(part)
    flush(cur_day, pending)

    np.save(os.path.join(a.out, 'nodes.npy'), nm.names(), allow_pickle=True)
    meta = dict(rows=n_rows, edges=n_edges, malicious_edges=n_mal, nodes=len(nm.d), users=len(um.d),
                window=a.window, redteam_events=len(rt), seconds=round(time.time() - t0))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    print(meta)


if __name__ == '__main__':
    main()
