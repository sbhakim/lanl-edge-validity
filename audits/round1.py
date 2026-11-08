"""Workload, no-learning diagnostics and rule-weight ablation (protocols/round1.txt, D-F).

Uses the saved scores only; no test label is used for fitting.

python round1.py ROOT OUT_JSON
"""
import json, os, sys
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from mln import ExactMLN, KNOWGRAPH_RULES

ROOT, OUTF = sys.argv[1:3]
R = os.path.join(ROOT, 'results'); PREP = os.path.join(ROOT, 'data', 'prep')
FPR, EVIL_W, SEEDS = 0.005, 150885 // 1800, range(5)
out = {}

P = dict(np.load(os.path.join(R, 'pairs.npz')))
fit, te = P['split'] == 0, P['split'] == 1
names = np.load(os.path.join(PREP, 'nodes.npy'), allow_pickle=True)
isC = np.array([str(n).startswith('C') for n in names])
src, dst, win, y = P['src'][te], P['dst'][te], P['win'][te], P['y'][te].astype(int)
src_n = names[src]; keep = src_n != 'C17693'
hosts = sorted(np.unique(src_n[y == 1]).tolist())
wb = float(P['n_test_benign_total']) / (y == 0).sum()          # inverse benign sampling rate

# full edge table
parts = []
for f in sorted(os.listdir(PREP)):
    if f.startswith('day_'):
        z = np.load(os.path.join(PREP, f))
        parts.append(pd.DataFrame({k: z[k] for k in ['win', 'src', 'dst', 'cnt', 'ntlm', 'nov_dst']}))
E = pd.concat(parts, ignore_index=True); del parts
E = E[(E.src != E.dst) & isC[E.src.values] & isC[E.dst.values]].reset_index(drop=True)
E['pair'] = (E.src.values.astype(np.int64) << 20) | E.dst.values
first_pair_win = E.groupby('pair')['win'].min()
E['new_edge'] = (E['win'].values == first_pair_win.reindex(E['pair'].values).values).astype(np.int8)
fan = E.groupby(['win', 'src']).size().rename('fanout').reset_index()
E = E.merge(fan, on=['win', 'src'], how='left')
fs = np.full(len(names), np.iinfo(np.int32).max, np.int64)
np.minimum.at(fs, E.src.values, E.win.values); np.minimum.at(fs, E.dst.values, E.win.values)
E['age_h'] = (E.win.values - fs[E.src.values]) * 0.5
test_days = (E.win.max() - EVIL_W + 1) / 48
out['test_days'] = float(test_days)

df = pd.DataFrame({'win': win, 'src': src, 'dst': dst}).merge(
    E[['win', 'src', 'dst', 'cnt', 'ntlm', 'nov_dst', 'new_edge', 'fanout', 'age_h']], on=['win', 'src', 'dst'], how='left')
assert df['cnt'].notna().all() and len(df) == len(y)


def diag_scores(d):
    ntlm = (d['ntlm'].values > 0).astype(float); first = (d['nov_dst'].values > 0).astype(float)
    return {'E1 NTLM': ntlm, 'E2 FirstAccess': first, 'E3 NewEdge': d['new_edge'].values.astype(float),
            'E4 FanOut': d['fanout'].values.astype(float), 'E5 HostAge': -np.log1p(d['age_h'].values),
            'E6 Rule (NTLM+FirstAccess)': ntlm + first}


def tp_at(yy, s, f=FPR):
    fp, tp, _ = roc_curve(yy, s)
    return float(np.interp(f, fp, tp))


def evaluate(s):
    thr = np.quantile(s[y == 0], 1 - FPR)
    alert = s > thr
    ben_alert = alert & (y == 0)
    o = {'auc_all': float(roc_auc_score(y, s)), 'tp_all': tp_at(y, s), 'tp_excl': tp_at(y[keep], s[keep]),
         'auc_excl': float(roc_auc_score(y[keep], s[keep])),
         'strict_tpr_all': float(alert[y == 1].mean()), 'strict_k_excl': int((alert & (y == 1) & keep).sum()),
         'benign_alerts_per_day_est': float(ben_alert.sum() * wb / test_days), 'hosts': {}}
    for h in hosts:
        m = (src_n == h) & (y == 1); hit = m & alert
        fw = int(win[m].min()); dw = int(win[hit].min()) if hit.any() else None
        o['hosts'][h] = {'k': int(hit.sum()), 'n': int(m.sum()), 'first_mal_win': fw, 'first_det_win': dw,
                         'delay_h': (dw - fw) * 0.5 if dw is not None else None,
                         'benign_alerts_before_detection_est': float((ben_alert & (win < dw)).sum() * wb) if dw is not None
                         else float((ben_alert & (win <= win[y == 1].max())).sum() * wb)}
    o['macro_host_coverage'] = float(np.mean([o['hosts'][h]['k'] > 0 for h in hosts]))
    o['macro_tpr'] = float(np.mean([o['hosts'][h]['k'] / o['hosts'][h]['n'] for h in hosts]))
    return o


# E: diagnostics on the test sample, and workload on all test edges
out['diagnostics'] = {k: evaluate(s) for k, s in diag_scores(df).items()}
Et = E[E.win >= EVIL_W]
full = diag_scores(Et)
day = (Et.win.values - EVIL_W) // 48
mal_idx = pd.MultiIndex.from_arrays([win[y == 1], src[y == 1], dst[y == 1]])
is_mal = pd.MultiIndex.from_arrays([Et.win.values, Et.src.values, Et.dst.values]).isin(mal_idx)
for k, s in full.items():
    thr = np.quantile(s[~is_mal], 1 - FPR); al = (s > thr) & ~is_mal
    per_day_hosts = pd.Series(Et.src.values[al]).groupby(day[al]).nunique()
    out['diagnostics'][k]['full_test'] = {
        'benign_alerts_per_day': float(al.sum() / test_days),
        'distinct_benign_src_hosts_alerted_per_day_mean': float(per_day_hosts.reindex(range(int(day.max()) + 1), fill_value=0).mean()),
        'realised_benign_fpr': float(al.sum() / (~is_mal).sum())}
del E, Et, full

# F: rule-weight ablation with bounded L-BFGS
def fit_bounded(preds, rules, Lf, yf, bounds_w, l2=1e-3):
    m = ExactMLN(preds, rules, 'main')
    nw, na = len(rules), len(preds)
    Lt = torch.tensor(Lf, dtype=torch.float32); yt = torch.tensor(yf, dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss()

    def f(x):
        xt = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        m.w.data = xt[:nw].detach().clone(); m.a.data = xt[nw:].detach().clone()
        m.zero_grad()
        loss = bce(m(Lt), yt) + l2 * (m.w.pow(2).sum() + (m.a - 1).pow(2).sum())
        loss.backward()
        return float(loss), np.r_[m.w.grad.numpy(), m.a.grad.numpy()].astype(np.float64)
    x0 = np.r_[[min(max(1.0, lo if lo is not None else -np.inf), hi if hi is not None else np.inf) for lo, hi in bounds_w], np.ones(na)]
    r = minimize(f, x0, jac=True, method='L-BFGS-B', bounds=list(bounds_w) + [(None, None)] * na, options={'maxiter': 500})
    m.w.data = torch.tensor(r.x[:nw], dtype=torch.float32); m.a.data = torch.tensor(r.x[nw:], dtype=torch.float32)
    return m, dict(zip(list(rules) + [f'scale_{p}' for p in preds], np.round(r.x, 3).tolist()))


preds = ['main', 'auth', 'encg']; R23 = {k: v for k, v in KNOWGRAPH_RULES.items() if k != 'R1_auth'}
cfgs = {'F1 learned R1-R3 (unconstrained)': (KNOWGRAPH_RULES, [(None, None)] * 3),
        'F2 learned R2-R3 (R1 removed)': (R23, [(None, None)] * 2),
        'F3 learned R1-R3, weights >= 0': (KNOWGRAPH_RULES, [(0, None)] * 3)}
for w in (0.5, 1, 2, 4):
    cfgs[f'F4 R1 fixed at {w:g}, R2-R3 learned'] = (KNOWGRAPH_RULES, [(w, w), (None, None), (None, None)])
seed_res = {}
for sd in SEEDS:
    ld = lambda n: np.load(os.path.join(R, f'{n}_scores_seed{sd}.npz'))['logit']
    ev = {'main': -ld('main'), 'auth': ld('auth'), 'encg': -ld('encg')}
    S = {'Main GNN': ev['main'][te], 'EncG': ev['encg'][te]}
    # reference fit, should match the KnowGraph re-implementation in analysis.json
    m0 = ExactMLN(preds, KNOWGRAPH_RULES, 'main')
    p0 = m0.fit(torch.tensor(np.stack([ev[p][fit] for p in preds], 1), dtype=torch.float32), torch.tensor(P['y'][fit]))
    with torch.no_grad():
        S['F0 original torch fit (reference)'] = m0(torch.tensor(np.stack([ev[p][te] for p in preds], 1), dtype=torch.float32)).numpy()
    params = {'F0 original torch fit (reference)': p0}
    for name, (rules, bw) in cfgs.items():
        pr = [p for p in preds if any(p == q for c in rules.values() for q, _ in c)]
        m, prm = fit_bounded(pr, rules, np.stack([ev[p][fit] for p in pr], 1), P['y'][fit], bw)
        with torch.no_grad():
            S[name] = m(torch.tensor(np.stack([ev[p][te] for p in pr], 1), dtype=torch.float32)).numpy()
        params[name] = prm
    seed_res[sd] = {k: evaluate(s) | {'params': params.get(k)} for k, s in S.items()}
    print('seed', sd, {k: round(v['tp_all'], 3) for k, v in seed_res[sd].items()}, flush=True)

agg = {}
for k in seed_res[0]:
    L = [seed_res[s][k] for s in SEEDS]
    a = {m: [round(d[m], 4) for d in L] for m in ('auc_all', 'tp_all', 'tp_excl', 'auc_excl', 'strict_tpr_all', 'strict_k_excl',
                                                  'benign_alerts_per_day_est', 'macro_host_coverage', 'macro_tpr')}
    a['hosts_k'] = {h: [d['hosts'][h]['k'] for d in L] for h in hosts}
    a['hosts_delay_h'] = {h: [d['hosts'][h]['delay_h'] for d in L] for h in hosts}
    a['benign_before_detection_est'] = {h: [round(d['hosts'][h]['benign_alerts_before_detection_est']) for d in L] for h in hosts}
    a['params'] = [d['params'] for d in L]
    agg[k] = a
out['per_seed_detectors'] = agg
json.dump(out, open(OUTF, 'w'), indent=1, default=float)
print('wrote', OUTF)
