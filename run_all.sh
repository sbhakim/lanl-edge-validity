#!/usr/bin/env bash
# Full pipeline, from the raw LANL files to every number in the paper.
# Expects data/auth.txt.gz and data/redteam.txt.gz. Takes about a day on one GPU.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd); PREP=data/prep; RES=results; SEEDS="0 1 2 3 4"
mkdir -p "$PREP" "$RES/logs"

python src/prep_lanl.py data/auth.txt.gz data/redteam.txt.gz "$PREP"
python src/make_pairs.py "$PREP" "$RES/pairs.npz" --holdout 4 --n_benign 2000000 --seed 0

for s in $SEEDS; do
  python src/main_model.py "$PREP" "$RES" --pairs "$RES/pairs.npz" --seed "$s" > "$RES/logs/main_s$s.log"
  python src/auth_model.py "$PREP" "$RES" --pairs "$RES/pairs.npz" --seed "$s" > "$RES/logs/auth_s$s.log"
  python src/encg.py       "$PREP" "$RES" --pairs "$RES/pairs.npz" --seed "$s" > "$RES/logs/encg_s$s.log"
  python src/variants.py "$RES/pairs.npz" "$RES" "$PREP" --seed "$s" --out "$RES/variants_s$s.json" > "$RES/logs/variants_s$s.log"
done
python src/analysis.py "$ROOT" > "$RES/logs/analysis.log"

# audits (protocols/)
python audits/prep_audit.py data/auth.txt.gz data/redteam.txt.gz "$RES/prep_audit"
python audits/round1.py "$ROOT" "$RES/round1.json"
for s in $SEEDS; do python audits/prospective.py "$ROOT" "$RES/prospective" --seed "$s"; done
python audits/round2.py "$ROOT" "$RES/round2.json"
python audits/encg_burst.py "$ROOT" "$RES/encg_burst.json"

# sanity check against the counts reported in the paper
python - "$ROOT" <<'EOF'
import json, sys
import numpy as np
root = sys.argv[1]
A = json.load(open(f'{root}/results/analysis.json'))
P = np.load(f'{root}/results/pairs.npz')
test = P['split'] == 1
checks = {
    'red-team events': (A['redteam']['events'], 749),
    'malicious test edges': (int(P['y'][test].sum()), 525),
    'C17693 malicious edges': (A['test']['mal_per_host']['C17693'], 504),
    'benign test edges': (int(P['n_test_benign_total']), 79904287),
    'sampled benign edges': (int((P['y'][test] == 0).sum()), 2000000),
    'fit pairs': (A['fit_pairs']['n'], 255624),
}
bad = 0
for name, (got, want) in checks.items():
    print(f"{'ok ' if got == want else 'BAD'} {name}: {got} (paper: {want})")
    bad += got != want
sys.exit(bad > 0)
EOF
