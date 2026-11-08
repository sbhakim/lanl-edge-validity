"""Knowledge-specified MLN: rule signs from domain knowledge, one shared weight w.

Nothing is fitted. The weight grid {0.5, 1, 2, 4} is reported in full.
Predicates: main and encg (negated logits), ntlm (observed NTLM share as smoothed
log-odds) and first (the edge contains a first-time account-to-destination access).
R4 encodes the ATT&CK step from valid accounts (T1078) to lateral movement (TA0008).

python variants.py PAIRS RESULTS DATA_PREP --seed 0 --out variants_s0.json
"""
import argparse, json, glob, os
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from mln import ExactMLN

RULES = {
    'R1': [('main', False), ('ntlm', True)],
    'R2': [('encg', False), ('main', True)],
    'R3': [('encg', True), ('main', False)],
    'R4': [('first', False), ('main', True)],
}


def metrics(y, s):
    f, t, _ = roc_curve(y, s)
    return dict(AUC=round(float(roc_auc_score(y, s)), 4), AP=round(float(average_precision_score(y, s)), 4),
                **{f'TP@{p}': round(float(np.interp(p / 100, f, t)), 4) for p in (0.5, 1, 2)})


def host_level(y, s, src, win, fpr=0.005):
    """For each attacker source host: detected at this FPR, and the delay in hours."""
    thr = np.quantile(s[y == 0], 1 - fpr)
    out = {}
    for h in np.unique(src[y == 1]):
        m = (src == h) & (y == 1)
        w0 = win[m].min()
        hit = m & (s > thr)
        out[str(h)] = dict(n_mal=int(m.sum()), detected=bool(hit.any()),
                           delay_h=(float((win[hit].min() - w0) * 0.5) if hit.any() else None),
                           frac_detected=round(float(hit.sum() / m.sum()), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs'); ap.add_argument('results'); ap.add_argument('prep')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    P = dict(np.load(a.pairs)); te = P['split'] == 1
    df = pd.DataFrame({'win': P['win'][te], 'src': P['src'][te], 'dst': P['dst'][te]})
    parts = []
    for f in sorted(glob.glob(os.path.join(a.prep, 'day_*.npz'))):
        z = np.load(f)
        parts.append(pd.DataFrame({k: z[k] for k in ['win', 'src', 'dst', 'cnt', 'ntlm', 'nov_dst']}))
    E = pd.concat(parts)
    df = df.merge(E, on=['win', 'src', 'dst'], how='left')
    assert df['cnt'].notna().all(), 'test pair not found in edge table'
    y = P['y'][te]
    names = np.load(os.path.join(a.prep, 'nodes.npy'), allow_pickle=True)
    src_n = names[P['src'][te]]; win = P['win'][te]
    ld = lambda n: np.load(f'{a.results}/{n}_scores_seed{a.seed}.npz')['logit'][te]
    ev = {'main': -ld('main'), 'encg': -ld('encg'),
          'ntlm': np.log((df['ntlm'].values + 0.5) / (df['cnt'].values - df['ntlm'].values + 0.5)),
          'first': np.where(df['nov_dst'].values > 0, 1.0, -1.0)}
    keep = src_n != 'C17693'
    res = {'n_test': int(te.sum()), 'n_mal': int(y.sum()), 'n_mal_excl': int(y[keep].sum()),
           'first_access_rate': {'malicious': round(float((df['nov_dst'].values[y == 1] > 0).mean()), 4),
                                 'benign': round(float((df['nov_dst'].values[y == 0] > 0).mean()), 4)},
           'ntlm_rate': {'malicious': round(float((df['ntlm'].values[y == 1] > 0).mean()), 4),
                         'benign': round(float((df['ntlm'].values[y == 0] > 0).mean()), 4)}}
    configs = {'main': (['main'], []), 'main+encg (R2,R3)': (['main', 'encg'], ['R2', 'R3']),
               'main+encg+ntlm (R1-3)': (['main', 'ntlm', 'encg'], ['R1', 'R2', 'R3']),
               'main+encg+first (R2-4)': (['main', 'encg', 'first'], ['R2', 'R3', 'R4']),
               'all (R1-4)': (['main', 'ntlm', 'encg', 'first'], ['R1', 'R2', 'R3', 'R4'])}
    for w in (0.5, 1.0, 2.0, 4.0):
        for name, (preds, rules) in configs.items():
            if name == 'main' and w != 1.0:
                continue
            m = ExactMLN(preds, {r: RULES[r] for r in rules}, 'main', init_w=w)
            L = torch.tensor(np.stack([ev[p] for p in preds], 1), dtype=torch.float32)
            with torch.no_grad():
                s = m(L).numpy()
            key = f'{name} | w={w}' if rules else name
            res[key] = {'all': metrics(y, s), 'excl_C17693': metrics(y[keep], s[keep]),
                        'hosts@0.5%FPR': host_level(y, s, src_n, win)}
    json.dump(res, open(a.out, 'w'), indent=1)
    for k, v in res.items():
        if isinstance(v, dict) and 'all' in v:
            print(f"{k:32s} TPR@0.5% {v['all']['TP@0.5']:.3f}  excl. C17693 {v['excl_C17693']['TP@0.5']:.3f}")


if __name__ == '__main__':
    main()
