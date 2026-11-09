# Never-Seen Edges Rival Knowledge-Enabled Detectors

Code for the paper *Never-Seen Edges Rival Knowledge-Enabled Detectors: A Measurement Study of Benchmark Validity in LANL Lateral-Movement Detection*.

We re-implement the LANL components of KnowGraph (Zhou et al., CCS 2024): an Euler-style main model, the authentication model, EncG, and a per-edge Markov logic network solved exactly. KnowGraph has no public code, so this is our own implementation, not the authors'. We then audit what the LANL labels reward: the red-team labels, twelve edge constructions, rule-weight ablations, alert budgets, prospective scoring and simple no-learning baselines. Each audit was written down in `protocols/` before it was run.

**Main finding.** 504 of the 525 malicious test edges come from one source host, C17693. Outside that host, no learned detector finds more than two of the remaining 21 edges, and a never-seen-edge flag with no training detects half of all attack edges, though none at 0.1% FPR and few outside C17693. A strong pooled LANL score is therefore not evidence of coverage across hosts.

## Repository layout

```
src/        preprocessing, models, MLN and the main analysis
audits/     the pre-specified audits and the reporting script
protocols/  audit protocols, fixed before the audits were run
run_all.sh  the full pipeline, from raw logs to final numbers
```

| Script | What it does |
|---|---|
| `src/prep_lanl.py` | 30-minute computer-to-computer edges, all authentication types |
| `src/make_pairs.py` | rule-fitting pairs and the shared test sample (all 525 malicious, 2 M benign edges) |
| `src/main_model.py`, `src/auth_model.py`, `src/encg.py` | neural predicates, one run per seed |
| `src/variants.py` | MLN with rule weights fixed by domain knowledge |
| `src/analysis.py` | detection metrics, per-host results and ROC curves |
| `audits/prep_audit.py` | label audit and the twelve edge constructions |
| `audits/round1.py`, `audits/round2.py` | workload, rule ablations, FPR budgets, new-edge baseline, bootstrap |
| `audits/prospective.py`, `audits/encg_burst.py` | prospective scoring and the EncG burst intervention |
| `audits/lanl_report.py` | the recommended reporting checks for any detector's edge scores |

## Setup

Python 3.10 with PyTorch 2.4 (CUDA 12.1), PyTorch Geometric 2.6, NumPy, SciPy, pandas and scikit-learn:

```bash
pip install "torch==2.4.*" --index-url https://download.pytorch.org/whl/cu121
pip install "torch_geometric==2.6.*" numpy scipy pandas scikit-learn
```

`audits/prep_audit.py` also needs `pigz`. We used one RTX 3090 (24 GB), 80 CPU cores and 314 GB of RAM.

The experiments use the public LANL authentication log and red-team file (CC0). Put `auth.txt.gz` and `redteam.txt.gz` in `data/`; [DATASET.md](DATASET.md) explains where to get them and how they are processed.

## Reproducing the results

```bash
bash run_all.sh
```

This takes about a day. It trains every model for five seeds, runs all audits, and ends by checking the recomputed counts against the paper: 749 red-team events, 525 malicious and 79,904,287 benign test edges, 504 malicious edges from C17693, and 255,624 rule-fitting pairs. Each script can also be run on its own; its arguments are listed at the top of the file.

Outputs go to `results/`:

| File | Contents |
|---|---|
| `analysis.json`, `roc_points.npz` | main detection results, per-host hits and low-FPR ROC curves |
| `prep_audit/prep_audit.json` | label audit and the twelve edge constructions |
| `round1.json`, `round2.json` | workload, rule ablations, FPR budgets, novelty baselines and bootstrap |
| `prospective/`, `encg_burst.json` | prospective scoring and the EncG burst intervention |
| `logs/` | one log per step |

## Checking your own detector

Save one score per test edge of `results/pairs.npz` (higher means more anomalous) as a NumPy array, then run

```bash
python audits/lanl_report.py . scores.npy
```

It prints per-host detections at a strict 0.5% FPR threshold, results with and without C17693, AP at the true prevalence, expected benign alerts per day, and the two no-learning baselines for comparison. It reads `results/` and `data/prep/nodes.npy`, so run `run_all.sh` first.

<!--
## Citation

```bibtex
@misc{hakim2026neverseen,
  title  = {Never-Seen Edges Rival Knowledge-Enabled Detectors: A Measurement Study of
            Benchmark Validity in {LANL} Lateral-Movement Detection},
  author = {Hakim, Safayat Bin and Adil, Muhammad and Song, Houbing Herbert},
  year   = {2026}
}
```

Please also cite the LANL dataset (Kent, 2015); its BibTeX entry is in [DATASET.md](DATASET.md).
-->

## License

MIT, see [LICENSE](LICENSE). Questions and bug reports are welcome as GitHub issues.
