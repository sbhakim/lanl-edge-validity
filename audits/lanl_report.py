"""Per-host LANL report for any detector's edge scores.

Applies the reporting checks recommended in the paper to a user-supplied score
vector on the shared test sample: pooled metrics, metrics without the dominant
source host, AP at the true prevalence, expected benign alerts per day, and
per-host detections at a strict 0.5% FPR threshold. Scores of the two
no-learning baselines (new edge, host age) are printed alongside for reference,
read from results/analysis.json.

Scores must be "higher = more anomalous", one per edge of results/pairs.npz,
either for all pairs or for the test edges only (split == 1, file order).

Usage: python lanl_report.py ROOT SCORES.npy
"""
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

DOMINANT = 'C17693'
BUDGETS = (0.001, 0.005, 0.01, 0.02)


def tpr_at(y, s, f):
    fp, tp, _ = roc_curve(y, s)
    return float(np.interp(f, fp, tp))


def main(root, score_path):
    P = np.load(os.path.join(root, 'results', 'pairs.npz'))
    te = P['split'] == 1
    s = np.load(score_path).astype(float).ravel()
    if len(s) == len(te):
        s = s[te]
    if len(s) != te.sum():
        sys.exit(f'expected {te.sum()} test scores (or {len(te)} for all pairs), got {len(s)}')
    y = P['y'][te].astype(int); win = P['win'][te]
    names = np.load(os.path.join(root, 'data', 'prep', 'nodes.npy'), allow_pickle=True)
    src = names[P['src'][te]]
    n_ben = int(P['n_test_benign_total'])
    days = (win.max() - win.min() + 1) * 0.5 / 24
    wb = n_ben / (y == 0).sum()  # inverse sampling weight of benign test edges
    keep = src != DOMINANT

    print(f'test sample: {y.sum()} malicious, {(y == 0).sum()} benign (of {n_ben}), {days:.1f} days')
    print(f"pooled   AUC {roc_auc_score(y, s):.3f}  AP at true prevalence "
          f"{average_precision_score(y, s, sample_weight=np.where(y == 1, 1.0, wb)):.2e}")
    print(f"without {DOMINANT} ({y[keep].sum()} edges)  AUC {roc_auc_score(y[keep], s[keep]):.3f}")
    for f in BUDGETS:
        print(f'  FPR {f:.1%}: TPR {tpr_at(y, s, f):.3f} pooled, {tpr_at(y[keep], s[keep], f):.3f} without '
              f'{DOMINANT}; about {f * n_ben / days:,.0f} benign alerts/day')

    thr = np.quantile(s[y == 0], 1 - 0.005)
    print('per source host at 0.5% FPR (strict threshold):')
    reached = 0
    for h in sorted(np.unique(src[y == 1])):
        m = (src == h) & (y == 1); hit = m & (s > thr)
        delay = f', first after {(win[hit].min() - win[m].min()) * 0.5:.0f} h' if hit.any() else ''
        reached += bool(hit.any())
        print(f'  {h}: {hit.sum()}/{m.sum()}{delay}')
    print(f'  hosts reached: {reached}/{len(np.unique(src[y == 1]))}')

    A = json.load(open(os.path.join(root, 'results', 'analysis.json')))['models']
    print('reference, no-learning baselines (mean of five seeds):')
    for k in ('New-edge baseline (never-seen src-dst pair)', 'Host-age baseline (new source host)'):
        a = A[k]
        n_reached = sum(v['k_min'] > 0 for v in a['hosts'].values())
        print(f"  {k.split(' (')[0]}: TPR@0.5% {a['tp_all']['mean']:.3f} pooled, "
              f"{a['tp_excl']['mean']:.3f} without {DOMINANT}; hosts reached {n_reached}/{len(a['hosts'])}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
