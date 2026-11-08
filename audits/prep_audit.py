"""Label audit and edge-construction variants (protocols/round1.txt, A and B).

One pass over auth.txt.gz. Counts edges and malicious edges under variants B0-B11 and
keeps every event that matches a red-team (time, source, destination) tuple.
Needs pigz.

python prep_audit.py AUTH_GZ REDTEAM_GZ OUT_DIR
"""
import json, os, subprocess, sys, time
import numpy as np
import pandas as pd

AUTH, RT, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
EVIL = 150885
COLS = ['ts', 'src_user', 'dst_user', 'src', 'dst', 'auth', 'logon', 'orient', 'ok']

rt_raw = pd.read_csv(RT, header=None, names=['ts', 'user', 'src', 'dst'])
rt = rt_raw[['ts', 'src', 'dst']].drop_duplicates().assign(label=np.int8(1))

# variant: (bucket in seconds, filters, keep self-loops, unit)
VARIANTS = {
    'B0': (1800, [], False, 'edge'), 'B1': (1800, ['ntlm'], False, 'edge'),
    'B2': (1800, ['logon'], False, 'edge'), 'B3': (1800, ['succ'], False, 'edge'),
    'B4': (1800, ['logon', 'succ'], False, 'edge'), 'B5': (1800, ['human'], False, 'edge'),
    'B6': (1800, ['logon', 'succ', 'human'], False, 'edge'), 'B7': (1800, [], True, 'edge'),
    'B8': (3600, [], False, 'edge'), 'B9': (60, [], False, 'edge'),
    'B10': (None, [], False, 'event'), 'B11': (1800, [], False, 'user'),
}
tot = {v: 0 for v in VARIANTS}; tst = {v: 0 for v in VARIANTS}
pos = {v: {} for v in VARIANTS}          # malicious key -> source host
nodes, users = {}, {}


def ids(s, d):
    new = pd.unique(s[~s.isin(d.keys())])
    for n in new:
        d[n] = len(d)
    return s.map(d).to_numpy(np.int64)


def count(df):
    """Count units in rows whose hour is complete."""
    for v, (b, filt, selfl, kind) in VARIANTS.items():
        m = df['isC'].to_numpy().copy()
        if not selfl:
            m &= df['src_i'].to_numpy() != df['dst_i'].to_numpy()
        for f in filt:
            m &= df[f].to_numpy()
        if kind == 'event':
            tot[v] += int(m.sum()); tst[v] += int((m & (df['ts'].to_numpy() >= EVIL)).sum())
            lab = m & (df['label'].to_numpy() == 1)
            for i, (t, s, d) in enumerate(zip(df['ts'].to_numpy()[lab], df['src'].to_numpy()[lab], df['dst'].to_numpy()[lab])):
                pos[v][(int(t), s, d, i)] = s
            continue
        bk = df['ts'].to_numpy()[m] // b
        a = (df['uid'] if kind == 'user' else df['src_i']).to_numpy()[m]
        key = (bk << 36) | (a << 18) | df['dst_i'].to_numpy()[m]
        u = np.unique(key)
        tot[v] += len(u); tst[v] += int(((u >> 36) >= EVIL // b).sum())
        lab = df['label'].to_numpy()[m] == 1
        for k, s in zip(key[lab].tolist(), df['src'].to_numpy()[m][lab]):
            pos[v][k] = s


def label_audit(rt_raw, ev):
    """Red-team file versus the auth events matched to it (protocol A1-A2)."""
    tup = rt_raw[['ts', 'src', 'dst']].drop_duplicates()
    hit = tup.merge(ev[['ts', 'src', 'dst']].drop_duplicates(), how='left', indicator=True)['_merge'] == 'both'
    users = rt_raw.groupby(['ts', 'src', 'dst'])['user'].agg(set)
    same_user = [u in users[(t, s, d)] for t, s, d, u in zip(ev['ts'], ev['src'], ev['dst'], ev['src_user'])]
    return {'rows': len(rt_raw), 'unique_rows': len(rt_raw.drop_duplicates()), 'tuples': len(tup),
            'rows_per_src': rt_raw['src'].value_counts().to_dict(),
            'tuples_matched': int(hit.sum()), 'unmatched_per_src': tup.loc[~hit.values, 'src'].value_counts().to_dict(),
            'matched_events': len(ev), **{c: ev[c].value_counts().to_dict() for c in ('auth', 'logon', 'orient', 'ok')},
            'redteam_user_is_source_user': int(sum(same_user))}


t0 = time.time(); n = 0; carry = None; matched = []
proc = subprocess.Popen(['pigz', '-dc', AUTH], stdout=subprocess.PIPE)
reader = pd.read_csv(proc.stdout, header=None, names=COLS, chunksize=20_000_000,
                     dtype={'ts': np.int64, **{c: str for c in COLS[1:]}})
for ch in reader:
    n += len(ch)
    ch = ch.merge(rt, on=['ts', 'src', 'dst'], how='left')
    ch['label'] = ch['label'].fillna(0).astype(np.int8)
    if ch['label'].any():
        matched.append(ch[ch['label'] == 1][COLS].copy())
    ch['src'] = ch['src'].str.replace('$', '', regex=False)
    ch['dst'] = ch['dst'].str.replace('$', '', regex=False)
    ch['isC'] = ch['src'].str.startswith('C').fillna(False).to_numpy() & ch['dst'].str.startswith('C').fillna(False).to_numpy()
    ch['ntlm'] = ch['auth'].fillna('').str.upper().str.contains('NTLM').to_numpy()
    ch['logon'] = (ch['orient'] == 'LogOn').to_numpy()
    ch['succ'] = (ch['ok'] == 'Success').to_numpy()
    su = ch['src_user'].fillna('')
    ch['human'] = (~su.str.contains('$', regex=False) & ~su.str.upper().str.startswith('ANONYMOUS')).to_numpy()
    ch['src_i'] = ids(ch['src'], nodes); ch['dst_i'] = ids(ch['dst'], nodes)
    ch['uid'] = ids(ch['src_user'].fillna('?'), users)
    ch = ch[['ts', 'src', 'dst', 'src_i', 'dst_i', 'uid', 'label', 'isC', 'ntlm', 'logon', 'succ', 'human']]
    if carry is not None:
        ch = pd.concat([carry, ch], ignore_index=True)
    last_h = ch['ts'].iloc[-1] // 3600
    hb = ch['ts'] // 3600
    carry = ch[hb == last_h]
    count(ch[hb < last_h])
    print(f'rows={n:,} {time.time()-t0:.0f}s', flush=True)
count(carry)

ev = pd.concat(matched) if matched else pd.DataFrame(columns=COLS)
ev.to_csv(os.path.join(OUT, 'redteam_matched_events.csv'), index=False)
res = {}
for v in VARIANTS:
    hosts = pd.Series(list(pos[v].values())).value_counts().to_dict() if pos[v] else {}
    npos = sum(hosts.values())
    p = np.array(list(hosts.values()), float) / max(npos, 1)
    res[v] = dict(spec=VARIANTS[v][:3] + (VARIANTS[v][3],), total=tot[v], test=tst[v], malicious=npos,
                  per_host=hosts, c17693_share=round(hosts.get('C17693', 0) / max(npos, 1), 4),
                  eff_hosts=round(1 / float((p ** 2).sum()), 3) if npos else None)
json.dump(dict(rows=n, seconds=round(time.time() - t0), labels=label_audit(rt_raw, ev), variants=res),
          open(os.path.join(OUT, 'prep_audit.json'), 'w'), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
