# Never-Seen Edges Rival Knowledge-Enabled Detectors

Code for *Never-Seen Edges Rival Knowledge-Enabled Detectors: A Measurement Study of Benchmark Validity in LANL Lateral-Movement Detection*.

We re-implement the LANL components of KnowGraph (Zhou et al., CCS 2024): an Euler-style main model, the authentication model, EncG, and a per-edge Markov logic network solved exactly. We then audit what the LANL labels reward, covering the red-team labels, twelve edge constructions, rule-weight ablations, alert budgets, prospective scoring and simple no-learning baselines. Each audit was written down in `protocols/` before it was run.

## Layout

```
src/        preprocessing, models, MLN and the main analysis
audits/     the pre-specified audits
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

## Requirements

Python 3.10, PyTorch 2.4 (CUDA 12.1), PyTorch Geometric 2.6, NumPy, SciPy, pandas and scikit-learn. `audits/prep_audit.py` also needs `pigz`. We used one RTX 3090 (24 GB), 80 CPU cores and 314 GB of RAM; the full pipeline takes about a day.

## Data

The experiments use the public LANL authentication log and red-team file. See [DATASET.md](DATASET.md) for where to get them and how to lay them out.

## Running

```bash
bash run_all.sh
```

The script finishes by checking the recomputed counts: 749 red-team events, 525 malicious and 79,904,287 benign test edges, 504 malicious edges from C17693, and 255,624 rule-fitting pairs. Every script can also be run on its own; its arguments are listed at the top of the file.

## Citation

```bibtex
@misc{hakim2026neverseen,
  title  = {Never-Seen Edges Rival Knowledge-Enabled Detectors: A Measurement Study of
            Benchmark Validity in {LANL} Lateral-Movement Detection},
  author = {Hakim, Safayat Bin and Song, Houbing Herbert},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
