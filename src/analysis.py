"""Main results: detection metrics, per-host hits and low-FPR ROC curves.

Reads data/redteam.txt.gz, data/prep, results/pairs.npz and the per-seed model scores.
Writes results/analysis.json and results/roc_points.npz. Learned rule weights are fitted
on the fit pairs only.

python analysis.py ROOT
"""
import json, os, sys
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
sys.path.insert(0, os.path.dirname(__file__))
from mln import ExactMLN, KNOWGRAPH_RULES
from data import DATE_OF_EVIL

ROOT = sys.argv[1]
R = os.path.join(ROOT, 'results'); PREP = os.path.join(ROOT, 'data', 'prep')
SEEDS = range(5)
FPR = 0.005
out = {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def tp_at(y, s, f=FPR):
    fp, tp, _ = roc_curve(y, s)
    return float(np.interp(f, fp, tp))


# red-team file
rt = pd.read_csv(os.path.join(ROOT, 'data', 'redteam.txt.gz'), header=None, names=['t', 'user', 'src', 'dst'])
out['redteam'] = {
    'events': len(rt), 'unique_events': int(len(rt.drop_duplicates())),
    'src_hosts': rt['src'].value_counts().to_dict(), 'n_dst': int(rt['dst'].nunique()), 'n_users': int(rt['user'].nunique()),
    'chained': int(len(set(rt['dst']) & set(rt['src']))),
    't_first': int(rt['t'].min()), 't_last': int(rt['t'].max()),
    'per_src': {h: {'n': int(len(g)), 't0_h': round((g['t'].min() - DATE_OF_EVIL) / 3600, 1),
                    't1_h': round((g['t'].max() - DATE_OF_EVIL) / 3600, 1), 'n_dst': int(g['dst'].nunique()),
                    'n_users': int(g['user'].nunique())} for h, g in rt.groupby('src')}}
rt[['t', 'src']].to_csv(os.path.join(R, 'redteam_times.csv'), index=False)

# test pairs and edge attributes
P = dict(np.load(os.path.join(R, 'pairs.npz')))
fit, te = P['split'] == 0, P['split'] == 1
names = np.load(os.path.join(PREP, 'nodes.npy'), allow_pickle=True)
src_n = names[P['src'][te]]; y = P['y'][te].astype(int); win = P['win'][te]
parts = []
for f in sorted(os.listdir(PREP)):
    if f.startswith('day_'):
        z = np.load(os.path.join(PREP, f))
        parts.append(pd.DataFrame({k: z[k] for k in ['win', 'src', 'dst', 'cnt', 'ntlm', 'nov_dst', 'nov_src']}))
E = pd.concat(parts, ignore_index=True)
df = pd.DataFrame({'win': win, 'src': P['src'][te], 'dst': P['dst'][te]}).merge(E, on=['win', 'src', 'dst'], how='left')
assert df['cnt'].notna().all()
ntlm = (df['ntlm'].values > 0).astype(float); first = (df['nov_dst'].values > 0).astype(float)
hosts = sorted(np.unique(src_n[y == 1]).tolist())
out['test'] = {'n': int(te.sum()), 'n_mal': int(y.sum()), 'n_benign_total': int(P['n_test_benign_total']),
               'mal_per_host': {h: int(((src_n == h) & (y == 1)).sum()) for h in hosts},
               'benign_from_attack_hosts': {h: int(((src_n == h) & (y == 0)).sum()) for h in hosts},
               'ntlm_rate': {'mal': float(ntlm[y == 1].mean()), 'ben': float(ntlm[y == 0].mean())},
               'first_rate': {'mal': float(first[y == 1].mean()), 'ben': float(first[y == 0].mean())},
               'first_rate_per_host': {h: float(first[(src_n == h) & (y == 1)].mean()) for h in hosts}}

# activity of the attacker hosts before the attack
D_c = E[['win', 'src', 'dst']]
evil_w = DATE_OF_EVIL // 1800
nid = {str(n): i for i, n in enumerate(names)}
out['attack_host_history'] = {}
for h in hosts:
    i = nid[h]
    pre = D_c[(D_c['win'] < evil_w) & ((D_c['src'] == i) | (D_c['dst'] == i))]
    post = D_c[(D_c['win'] >= evil_w) & (D_c['src'] == i)]
    out['attack_host_history'][h] = {'edges_before_attack_any_dir': int(len(pre)), 'out_edges_test_period': int(len(post)),
                                     'first_seen_h': (round((pre['win'].min() * 1800 - DATE_OF_EVIL) / 3600, 1)
                                                      if len(pre) else None)}
# host age: hours since the source host first appeared in the graph
isC = np.array([str(n).startswith('C') for n in names])
cc = E[(E['src'] != E['dst']) & isC[E['src'].values] & isC[E['dst'].values]]
fs = np.full(len(names), np.iinfo(np.int32).max, np.int64)
np.minimum.at(fs, cc['src'].values, cc['win'].values); np.minimum.at(fs, cc['dst'].values, cc['win'].values)
age_h = (win - fs[P['src'][te]]) * 0.5
assert (age_h >= 0).all()
# new edge: the (src, dst) pair occurs for the first time in this window
pk = (cc['src'].values.astype(np.int64) << 20) | cc['dst'].values
first_pair = pd.Series(cc['win'].values).groupby(pk).min()
te_pk = (P['src'][te].astype(np.int64) << 20) | P['dst'][te]
new_edge = (win == first_pair.reindex(te_pk).values).astype(float)
assert not np.isnan(first_pair.reindex(te_pk).values).any()
out['new_edge'] = {'mal_rate': float(new_edge[y == 1].mean()), 'ben_rate': float(new_edge[y == 0].mean()),
                   'per_host': {h: int(new_edge[(src_n == h) & (y == 1)].sum()) for h in hosts}}
seen_before = fs < evil_w
out['host_age'] = {'test_benign_src_unseen_before_attack': float((~seen_before[P['src'][te]])[y == 0].mean()),
                   'test_mal_src_unseen_before_attack': float((~seen_before[P['src'][te]])[y == 1].mean()),
                   'nodes_first_seen_in_test_period': int(((fs >= evil_w) & (fs < np.iinfo(np.int32).max)).sum()),
                   'nodes_seen_before_attack': int(seen_before.sum())}
del E, parts, cc

# detectors, per seed
keep = src_n != 'C17693'
rule = ntlm + first
KRULES = {'R1': [('main', False), ('ntlm', True)], 'R2': [('encg', False), ('main', True)],
          'R3': [('encg', True), ('main', False)], 'R4': [('first', False), ('main', True)]}


def per_host(s):
    thr = np.quantile(s[y == 0], 1 - FPR)
    o = {}
    for h in hosts:
        m = (src_n == h) & (y == 1)
        hit = m & (s > thr)
        o[h] = {'k': int(hit.sum()), 'n': int(m.sum()),
                'delay_h': float((win[hit].min() - win[m].min()) * 0.5) if hit.any() else None,
                'auc': float(roc_auc_score(np.r_[np.ones(m.sum()), np.zeros((y == 0).sum())],
                                           np.r_[s[m], s[y == 0]]))}
    return o


def summarize(s):
    wb = P['n_test_benign_total'] / (y == 0).sum()          # benign edges are subsampled, so reweight for AP
    aps = lambda yy, ss: average_precision_score(yy, ss, sample_weight=np.where(yy == 1, 1.0, wb))
    return {'ap_all': float(aps(y, s)), 'tp1_all': tp_at(y, s, .01), 'tp2_all': tp_at(y, s, .02),
            'ap_excl': float(aps(y[keep], s[keep])), 'tp1_excl': tp_at(y[keep], s[keep], .01),
            'tp2_excl': tp_at(y[keep], s[keep], .02),
            'auc_all': float(roc_auc_score(y, s)), 'tp_all': tp_at(y, s),
            'auc_excl': float(roc_auc_score(y[keep], s[keep])), 'tp_excl': tp_at(y[keep], s[keep]),
            'hosts': per_host(s)}


models = {}
inv = []
roc = {}
grid_x = np.r_[0, np.logspace(-4, np.log10(0.05), 200)]
for sd in SEEDS:
    ld = lambda n: np.load(os.path.join(R, f'{n}_scores_seed{sd}.npz'))['logit']
    ev = {'main': -ld('main'), 'auth': ld('auth'), 'encg': -ld('encg')}
    # KnowGraph re-implementation: rule weights fitted on the fit pairs
    preds = ['main', 'auth', 'encg']
    m = ExactMLN(preds, KNOWGRAPH_RULES, 'main')
    Lf = torch.tensor(np.stack([ev[p][fit] for p in preds], 1), dtype=torch.float32)
    params = m.fit(Lf, torch.tensor(P['y'][fit]))
    with torch.no_grad():
        kg = m(torch.tensor(np.stack([ev[p][te] for p in preds], 1), dtype=torch.float32)).numpy()
    # why R1 flips sign: NTLM evidence on proxy anomalies, observed edges and real attacks
    pa = 1 / (1 + np.exp(-ev['auth']))
    inv.append({'R1_weight': params['R1_auth'], 'p_ntlm_fit_observed': float(pa[fit][P['y'][fit] == 0].mean()),
                'p_ntlm_fit_proxy_anomaly': float(pa[fit][P['y'][fit] == 1].mean()),
                'p_ntlm_test_attack': float(pa[te][y == 1].mean()), 'p_ntlm_test_benign': float(pa[te][y == 0].mean())})
    evk = {'main': ev['main'][te], 'encg': ev['encg'][te],
           'ntlm': np.log((df['ntlm'].values + 0.5) / (df['cnt'].values - df['ntlm'].values + 0.5)),
           'first': np.where(first > 0, 1.0, -1.0)}
    m2 = ExactMLN(['main', 'encg'], {k: v for k, v in KNOWGRAPH_RULES.items() if k != 'R1_auth'}, 'main')
    m2.fit(torch.tensor(np.stack([ev[p][fit] for p in ['main', 'encg']], 1), dtype=torch.float32), torch.tensor(P['y'][fit]))
    with torch.no_grad():
        me = m2(torch.tensor(np.stack([ev[p][te] for p in ['main', 'encg']], 1), dtype=torch.float32)).numpy()
    inv[-1]['params_full'] = params
    S = {'Main GNN (Euler arch.)': ev['main'][te], 'EncG (SEAL)': ev['encg'][te], 'Learned MLN main+EncG (R2,R3)': me,
         'KnowGraph re-impl. (learned MLN)': kg}
    for w in (1.0, 2.0, 4.0):
        km = ExactMLN(['main', 'ntlm', 'encg', 'first'], KRULES, 'main', init_w=w)
        with torch.no_grad():
            S[f'Knowledge MLN R1-R4 (w={w:g})'] = km(torch.tensor(np.stack(
                [evk[p] for p in ['main', 'ntlm', 'encg', 'first']], 1), dtype=torch.float32)).numpy()
    S['Rule baseline (NTLM + first access)'] = rule
    S['Host-age baseline (new source host)'] = -np.log1p(age_h)
    S['New-edge baseline (never-seen src-dst pair)'] = new_edge
    for k, s in S.items():
        models.setdefault(k, []).append(summarize(s))
        for tag, mask in (('all', np.ones_like(keep)), ('excl', keep)):
            fp, tp, _ = roc_curve(y[mask], s[mask])
            roc.setdefault(f'{k}|{tag}', []).append(np.interp(grid_x, fp, tp))

agg = {}
for k, L in models.items():
    a = {}
    for met in ('auc_all', 'ap_all', 'tp_all', 'tp1_all', 'tp2_all', 'auc_excl', 'ap_excl', 'tp_excl', 'tp1_excl', 'tp2_excl'):
        x = np.array([d[met] for d in L])
        a[met] = {'mean': float(x.mean()), 'ci95': float(stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))),
                  'min': float(x.min()), 'max': float(x.max())}
    a['hosts'] = {h: {'k_mean': float(np.mean([d['hosts'][h]['k'] for d in L])), 'n': L[0]['hosts'][h]['n'],
                      'k_min': int(min(d['hosts'][h]['k'] for d in L)), 'k_max': int(max(d['hosts'][h]['k'] for d in L)),
                      'auc_mean': float(np.mean([d['hosts'][h]['auc'] for d in L])),
                      'delay_h': [d['hosts'][h]['delay_h'] for d in L]} for h in hosts}
    kx = [round(d['tp_excl'] * 21) for d in L]
    a['excl_hits_at_0.5'] = kx
    a['excl_wilson_of_max'] = wilson(max(kx), int(y[keep].sum()))
    agg[k] = a
out['models'] = agg
out['rule_inversion'] = inv
out['fit_pairs'] = {'n': int(fit.sum()), 'holdout_windows': P['holdout_windows'].tolist()}
np.savez_compressed(os.path.join(R, 'roc_points.npz'), x=grid_x,
                    **{k.replace('|', '__'): np.array(v) for k, v in roc.items()})
json.dump(out, open(os.path.join(R, 'analysis.json'), 'w'), indent=1, default=float)
for k, a in agg.items():
    print(f"{k:42s} AUC {a['auc_all']['mean']:.3f}±{a['auc_all']['ci95']:.3f} TP {a['tp_all']['mean']:.3f}±{a['tp_all']['ci95']:.3f} "
          f"| excl AUC {a['auc_excl']['mean']:.3f} TP {a['tp_excl']['mean']:.3f} hits {a['excl_hits_at_0.5']}")
    print('    hosts', {h: (round(v['k_mean'], 1), v['n'], round(v['auc_mean'], 3)) for h, v in a['hosts'].items()})
