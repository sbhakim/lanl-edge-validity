# Dataset

We use the **LANL Comprehensive, Multi-Source Cyber-Security Events** dataset (Kent, 2015), which covers 58 consecutive days of activity on the Los Alamos National Laboratory enterprise network. It is public under CC0.

- Download page: https://csr.lanl.gov/data/cyber1/
- DOI: [10.17021/1179829](https://doi.org/10.17021/1179829)

Only two of its five files are needed:

| File | Size (gzip) | Content |
|---|---|---|
| `auth.txt.gz` | 7.6 GB | 1,051,430,459 authentication events |
| `redteam.txt.gz` | 5 KB | 749 red-team compromise events |

The `dns`, `flows` and `proc` files are not used.

## Format

Both files are comma-separated, without a header, and ordered by time. Time is in seconds from the start of the collection.

`auth.txt.gz`: one authentication event per line.

```
time,source user@domain,destination user@domain,source computer,destination computer,
authentication type,logon type,authentication orientation,success/failure
1,ANONYMOUS LOGON@C586,ANONYMOUS LOGON@C586,C1250,C586,NTLM,Network,LogOn,Success
```

`redteam.txt.gz`: authentication events that are part of a red-team exercise.

```
time,user@domain,source computer,destination computer
150885,U620@DOM1,C17693,C1003
```

The red-team file lists only part of the exercise. All of its events come from four source computers, and C17693 accounts for 701 of the 749. The first event (second 150,885) splits training from testing.

## Setup

Place both files, unchanged, in `data/`:

```
data/
  auth.txt.gz
  redteam.txt.gz
```

`run_all.sh` writes the preprocessed edges to `data/prep/` (about 1 h, ~250 MB) and everything else to `results/`. Neither directory is tracked by git.

## Notes on preprocessing

- Events are grouped into 30-minute windows. Each (source, destination, window) becomes one directed edge, keeping its event count and number of NTLM events.
- We keep computer-to-computer edges without self-loops, and all authentication types. This gives 82.1 M edges, 525 of them malicious.
- An edge is malicious if a red-team event matches its time, source and destination exactly.
- `audits/prep_audit.py` repeats the counts under twelve alternative constructions, for example NTLM only, LogOn only, successful events only, no machine accounts, 1-minute and 1-hour windows, raw events, and account-level edges.

## Citation

```bibtex
@misc{kent2015lanl,
  author       = {Kent, Alexander D.},
  title        = {Comprehensive, Multi-Source Cyber-Security Events},
  howpublished = {Los Alamos National Laboratory},
  year         = {2015},
  doi          = {10.17021/1179829}
}
```
