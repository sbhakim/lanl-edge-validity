"""Novelty scores, FPR budgets, paired R1 audit, R5 and the benign bootstrap
(protocols/round2.txt, R2-1, R2-2, R2-4 to R2-6). Uses the saved scores only.

python round2.py ROOT OUT_JSON
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from mln import ExactMLN, KNOWGRAPH_RULES

ROOT, OUTF = sys.argv[1:3]
R = os.path.join(ROOT, 'results'); PREP = os.path.join(ROOT, 'data', 'prep')
EVIL_W, SEEDS, t0 = 150885 // 1800, range(5), time.time()
KRULES = {'R1': [('main', False), ('ntlm', True)], 'R2': [('encg', False), ('main', True)],
          'R3': [('encg', True), ('main', False)], 'R4': [('first', False), ('main', True)]}
out = {}
P = dict(np.load(os.path.join(R, 'pairs.npz')))
fit, te = P['split'] == 0, P['split'] == 1
names = np.load(os.path.join(PREP, 'nodes.npy'), allow_pickle=True)
isC = np.array([str(n).startswith('C') for n in names])
src, dst, win, y = P['src'][te], P['dst'][te], P['win'][te], P['y'][te].astype(int)
src_n = names[src]; keep = src_n != 'C17693'
hosts = ['C17693', 'C19932', 'C22409', 'C18025']
wb = float(P['n_test_benign_total']) / (y == 0).sum()

# edge table and novelty features
parts = []
for f in sorted(os.listdir(PREP)):
    if f.startswith('day_'):
        z = np.load(os.path.join(PREP, f))
        parts.append(pd.DataFrame({k: z[k] for k in ['win', 'src', 'dst', 'cnt', 'ntlm', 'nov_dst']}))
E = pd.concat(parts, ignore_index=True); del parts
E = E[(E.src != E.dst) & isC[E.src.values] & isC[E.dst.values]].reset_index(drop=True)
s64, d64 = E.src.values.astype(np.int64), E.dst.values.astype(np.int64)
E['dkey'] = (s64 << 20) | d64
E['ukey'] = (np.minimum(s64, d64) << 20) | np.maximum(s64, d64)
for k, col in (('dkey', 'N1'), ('ukey', 'N2')):
    fw = E.groupby(k)['win'].min()
    E[col] = (E['win'].values == fw.reindex(E[k].values).values).astype(np.int8)
fs = np.full(len(names), np.iinfo(np.int32).max, np.int64)
np.minimum.at(fs, E.src.values, E.win.values); np.minimum.at(fs, E.dst.values, E.win.values)
E['N3'] = (E.win.values == fs[E.src.values]).astype(np.int8)
E['age_h'] = (E.win.values - fs[E.src.values]) * 0.5
test_days = (E.win.max() - EVIL_W + 1) / 48
df = pd.DataFrame({'win': win, 'src': src, 'dst': dst}).merge(
    E[['win', 'src', 'dst', 'cnt', 'ntlm', 'nov_dst', 'N1', 'N2', 'N3', 'age_h']], on=['win', 'src', 'dst'], how='left')
assert df['cnt'].notna().all() and len(df) == len(y)
print('features', round(time.time() - t0), flush=True)


def tp_at(yy, s, f):
    fp, tp, _ = roc_curve(yy, s); return float(np.interp(f, fp, tp))


def host_view(s, thr):
    al = s > thr; ben_al = al & (y == 0)
    o = {}
    for h in hosts:
        m = (src_n == h) & (y == 1); hit = m & al
        fw = int(win[m].min()); dw = int(win[hit].min()) if hit.any() else None
        o[h] = {'k': int(hit.sum()), 'n': int(m.sum()), 'delay_h': (dw - fw) * .5 if dw is not None else None,
                'benign_alerts_before_first_hit_est': round(float((ben_al & (win < dw)).sum() * wb)) if dw is not None else 'never'}
    return o


def evaluate(s, fpr=0.005):
    thr = np.quantile(s[y == 0], 1 - fpr); al = s > thr
    return {'tp_interp_all': tp_at(y, s, fpr), 'tp_interp_excl': tp_at(y[keep], s[keep], fpr),
            'tp_strict_all': float(al[y == 1].mean()), 'k_strict_excl': int((al & (y == 1) & keep).sum()),
            'tp_strict_excl': float(al[(y == 1) & keep].mean()),
            'realised_fpr': float(al[y == 0].mean()), 'auc_all': float(roc_auc_score(y, s)),
            'auc_excl': float(roc_auc_score(y[keep], s[keep])), 'hosts': host_view(s, thr)}


# R2-1 novelty decomposition
out['R2_1'] = {}
Et = E[E.win >= EVIL_W]; day = (Et.win.values - EVIL_W) // 48
mal_idx = pd.MultiIndex.from_arrays([win[y == 1], src[y == 1], dst[y == 1]])
is_mal = pd.MultiIndex.from_arrays([Et.win.values, Et.src.values, Et.dst.values]).isin(mal_idx)
for k in ('N1', 'N2', 'N3'):
    o = evaluate(df[k].values.astype(float))
    al = (Et[k].values == 1) & ~is_mal            # binary score: an alert whenever the flag is set
    ph = pd.Series(Et.src.values[al]).groupby(day[al]).nunique().reindex(range(int(day.max()) + 1), fill_value=0)
    o['full_test'] = {'benign_edge_alerts_per_day': float(al.sum() / test_days), 'distinct_benign_src_per_day': float(ph.mean()),
                      'realised_benign_fpr': float(al.sum() / (~is_mal).sum())}
    out['R2_1'][k] = o
del Et
print('R2-1 done', round(time.time() - t0), flush=True)


# bounded MLN fit
def fit_bounded(preds, rules, Lf, yf, bw, ba=None, l2=1e-3):
    m = ExactMLN(preds, rules, 'main'); nw, na = len(rules), len(preds)
    Lt, yt = torch.tensor(Lf, dtype=torch.float32), torch.tensor(yf, dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss()

    def f(x):
        m.w.data = torch.tensor(x[:nw], dtype=torch.float32); m.a.data = torch.tensor(x[nw:], dtype=torch.float32)
        m.zero_grad()
        loss = bce(m(Lt), yt) + l2 * (m.w.pow(2).sum() + (m.a - 1).pow(2).sum()); loss.backward()
        return float(loss), np.r_[m.w.grad.numpy(), m.a.grad.numpy()].astype(np.float64)
    ba = ba or [(None, None)] * na
    x0 = np.r_[[lo if (lo is not None and lo == hi) else (max(lo, 1.0) if lo is not None else 1.0) for lo, hi in bw],
               [max(lo, 1.0) if lo is not None else 1.0 for lo, hi in ba]]
    r = minimize(f, x0, jac=True, method='L-BFGS-B', bounds=list(bw) + list(ba), options={'maxiter': 500})
    m.w.data = torch.tensor(r.x[:nw], dtype=torch.float32); m.a.data = torch.tensor(r.x[nw:], dtype=torch.float32)
    _, g = f(r.x)
    bnds = list(bw) + list(ba); pg = g.copy()
    for i, (lo, hi) in enumerate(bnds):                     # projected gradient: zero at active bounds
        if lo is not None and hi is not None and lo == hi: pg[i] = 0
        elif lo is not None and abs(r.x[i] - lo) < 1e-8 and g[i] > 0: pg[i] = 0
        elif hi is not None and abs(r.x[i] - hi) < 1e-8 and g[i] < 0: pg[i] = 0
    diag = {'success': bool(r.success), 'message': str(r.message), 'objective': round(float(r.fun), 6),
            'grad_norm_free': float(np.linalg.norm(pg)), 'nit': int(r.nit)}
    return m, dict(zip(list(rules) + [f'scale_{p}' for p in preds], np.round(r.x, 3).tolist())), diag


def run(m, L):
    with torch.no_grad():
        return m(torch.tensor(L, dtype=torch.float32)).numpy()


preds = ['main', 'auth', 'encg']; R23 = {k: v for k, v in KNOWGRAPH_RULES.items() if k != 'R1_auth'}
ntlm_lo = np.log((df['ntlm'].values + 0.5) / (df['cnt'].values - df['ntlm'].values + 0.5))
first_lo = np.where(df['nov_dst'].values > 0, 1.0, -1.0)
ne_lo = np.where(df['N1'].values > 0, np.log(99), -np.log(99))
rule = (df['ntlm'].values > 0).astype(float) + (df['nov_dst'].values > 0)
hostage = -np.log1p(df['age_h'].values)
BUDGETS = (0.0005, 0.001, 0.005, 0.01)
out['R2_2'], out['R2_4'], out['R2_5'] = {}, {}, {}
boot_scores = {}
for sd in SEEDS:
    ld = lambda n: np.load(os.path.join(R, f'{n}_scores_seed{sd}.npz'))['logit']
    ev = {'main': -ld('main'), 'auth': ld('auth'), 'encg': -ld('encg')}
    Lfit = lambda pr: np.stack([ev[p][fit] for p in pr], 1)
    Lte = lambda pr: np.stack([ev[p][te] for p in pr], 1)
    # learned fit, kept frozen for R2-5
    m0 = ExactMLN(preds, KNOWGRAPH_RULES, 'main'); m0.fit(torch.tensor(Lfit(preds), dtype=torch.float32), torch.tensor(P['y'][fit]))
    kg = run(m0, Lte(preds))
    km4 = ExactMLN(['main', 'ntlm', 'encg', 'first'], KRULES, 'main', init_w=4.0)
    Lk = np.stack([ev['main'][te], ntlm_lo, ev['encg'][te], first_lo], 1)
    kmln = run(km4, Lk)
    S = {'Main GNN': ev['main'][te], 'EncG': ev['encg'][te], 'KnowGraph re-impl.': kg, 'Knowledge MLN w=4': kmln,
         'Rule': rule, 'HostAge': hostage, 'N1 NewEdge': df['N1'].values.astype(float)}
    # R2-2 budgets and host-exclusion thresholds
    for k, s in S.items():
        if sd > 0 and k in ('Rule', 'HostAge', 'N1 NewEdge'):
            continue
        o = {f'{b * 100:g}%': evaluate(s, b) for b in BUDGETS}
        thr_pool = np.quantile(s[y == 0], .995); ex = {}
        for h in hosts:
            bm = (y == 0) & (src_n != h)
            thr_h = np.quantile(s[bm], .995); m = (src_n == h) & (y == 1)
            ex[h] = {'k_pooled': int((s[m] > thr_pool).sum()), 'k_excl_own_benign': int((s[m] > thr_h).sum()),
                     'n_own_benign_in_sample': int(((y == 0) & (src_n == h)).sum())}
        o['host_exclusion_0.5%'] = ex
        out['R2_2'].setdefault(k, {})[sd] = o
    if sd == 0:
        boot_scores = dict(S)
    # R2-4 paired R1 audit and F4b
    cf = {'F1': (preds, KNOWGRAPH_RULES, [(None, None)] * 3, None), 'F2': (['main', 'encg'], R23, [(None, None)] * 2, None),
          'F3': (preds, KNOWGRAPH_RULES, [(0, None)] * 3, None)}
    for w in (0.5, 1, 2, 4):
        cf[f'F4 R1={w:g}'] = (preds, KNOWGRAPH_RULES, [(w, w), (None, None), (None, None)], None)
        cf[f'F4b R1={w:g}, scale_auth>=0'] = (preds, KNOWGRAPH_RULES, [(w, w), (None, None), (None, None)],
                                              [(None, None), (0, None), (None, None)])
    res4 = {}
    for k, (pr, rules, bw, ba) in cf.items():
        m, prm, dg = fit_bounded(pr, rules, Lfit(pr), P['y'][fit], bw, ba)
        s = run(m, Lte(pr)); e = evaluate(s)
        res4[k] = {'params': prm, 'diag': dg, 'tp_interp_all': e['tp_interp_all'], 'tp_strict_all': e['tp_strict_all'],
                   'k_strict_excl': e['k_strict_excl'], 'hosts_k': {h: e['hosts'][h]['k'] for h in hosts}, '_s': s}
    a1, a2 = res4['F1']['_s'], res4['F2']['_s']
    t1, t2 = np.quantile(a1[y == 0], .995), np.quantile(a2[y == 0], .995)
    h1, h2 = (a1 > t1) & (y == 1), (a2 > t2) & (y == 1)
    res4['paired_F1_vs_F2'] = {'gained_by_F1': {h: int((h1 & ~h2 & (src_n == h)).sum()) for h in hosts},
                               'lost_by_F1': {h: int((~h1 & h2 & (src_n == h)).sum()) for h in hosts},
                               'tp_interp_diff': res4['F1']['tp_interp_all'] - res4['F2']['tp_interp_all']}
    for v in res4.values():
        v.pop('_s', None)
    out['R2_4'][sd] = res4
    # R2-5 new edge as an extra rule, no refitting
    r5 = {}
    for w5 in (0.5, 1, 2, 4):
        ma = ExactMLN(preds + ['ne'], dict(KNOWGRAPH_RULES, R5=[('ne', False), ('main', True)]), 'main')
        ma.w.data = torch.cat([m0.w.data, torch.tensor([float(w5)])]); ma.a.data = torch.cat([m0.a.data, torch.tensor([1.0])])
        sa = run(ma, np.c_[Lte(preds), ne_lo]); ea = evaluate(sa)
        mb = ExactMLN(['main', 'ntlm', 'encg', 'first', 'ne'], dict(KRULES, R5=[('ne', False), ('main', True)]), 'main', init_w=4.0)
        mb.w.data[-1] = float(w5)
        sb = run(mb, np.c_[Lk, ne_lo]); eb = evaluate(sb)
        for tag, e in (('KnowGraph re-impl. + R5', ea), ('Knowledge MLN w=4 + R5', eb)):
            r5[f'{tag} (w5={w5:g})'] = {k2: e[k2] for k2 in ('tp_interp_all', 'tp_strict_all', 'k_strict_excl', 'tp_interp_excl', 'auc_all')} | \
                                        {'hosts_k': {h: e['hosts'][h]['k'] for h in hosts}}
    for tag, s in (('KnowGraph re-impl. (no R5)', kg), ('Knowledge MLN w=4 (no R5)', kmln), ('N1 alone', ne_lo)):
        e = evaluate(s)
        r5[tag] = {k2: e[k2] for k2 in ('tp_interp_all', 'tp_strict_all', 'k_strict_excl', 'tp_interp_excl', 'auc_all')} | \
                  {'hosts_k': {h: e['hosts'][h]['k'] for h in hosts}}
    out['R2_5'][sd] = r5
    print('seed', sd, round(time.time() - t0), flush=True)

# R2-6 bootstrap of the benign sample (seed 0)
rng = np.random.default_rng(0); B = 1000
nb = int((y == 0).sum()); kq = int(np.ceil(.995 * (nb - 1)))
ben = {k: s[y == 0] for k, s in boot_scores.items()}
pos = {k: {h: np.sort(s[(src_n == h) & (y == 1)]) for h in hosts} for k, s in boot_scores.items()}
acc = {k: {'thr': [], 'tpr': [], **{h: [] for h in hosts}} for k in boot_scores}
for b in range(B):
    idx = rng.integers(0, nb, nb)
    for k in boot_scores:
        thr = np.quantile(ben[k][idx], .995)
        ks = {h: int(len(pos[k][h]) - np.searchsorted(pos[k][h], thr, 'right')) for h in hosts}
        acc[k]['thr'].append(float(thr)); acc[k]['tpr'].append(sum(ks.values()) / 525)
        for h in hosts:
            acc[k][h].append(ks[h])
    if b % 100 == 0:
        print('boot', b, round(time.time() - t0), flush=True)
out['R2_6'] = {k: {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 50)), float(np.percentile(v, 97.5))]
                   for m, v in a.items()} for k, a in acc.items()}
json.dump(out, open(OUTF, 'w'), indent=1, default=float)
print('wrote', OUTF, round(time.time() - t0))
