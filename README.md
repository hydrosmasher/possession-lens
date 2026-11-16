# Possession Lens

Frame-by-frame home, away and dead-ball possession from automatic football tracking, with a local interactive dashboard. Built by Hrishikesh Karmalkar, Basel, Omar, Michail and Paul. Registered team: **Visca el Barça, visca Catalunya**.

## Reproduce the submission

The frozen submission uses **candidate 053**. Obtain the licensed challenge data separately; never commit it. Use Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gru-inference-lock.txt
python -m controlswitch.recurrent_pipeline --data-root /path/to/provided-data --out local/submission-053 --config reports/external/short-spell-053-config.json --filename "Visca el Barça, visca Catalunya.csv"
python evaluation/official/evaluation/validate_submission.py "local/submission-053/Visca el Barça, visca Catalunya.csv"
```

On Windows, activate with `.venv\Scripts\activate`. For Linux/Windows CPU-only installation, first install `torch==2.9.1` from `https://download.pytorch.org/whl/cpu`, then install the pinned requirements. No GPU or hosted service is needed.

Expected CSV: **199,319 data rows**, columns `frame,label`, H/A/D inside the published halves and X outside. Expected SHA-256:

```text
2f5a445d2c3c10b3126b5bda3a5b9b11050d9a87410f19b7cbc5aaa88b928eeb
```

The latest source reproduced this exact file on16 November2026. The official format check does not measure accuracy. The pipeline also writes model-estimated statistics/timeline, provenance and observation-quality flags; these flags are not calibrated confidence or verified events.

## Open the website

Open [dashboard.html](dashboard.html), or double-click **Open Possession Lens.command** on macOS. Bundled named external examples work without installation, internet or a server. Choose a match, inspect its possession timeline, and compare provider-mapped players. Unresolved player control is kept visible. Verified passes/shots/substitutions appear only for the example with licensed provider events; touches are not fabricated.

The gallery uses its separately frozen model. Gallery statistics do not imply that candidate053 was rerun on those matches. The challenge tracking has slots rather than verified player identities.

## Open the notebook

```sh
python -m pip install -r requirements-gru-notebook.txt
python -m ipykernel install --user --name possession-lens-gru-312 --display-name "Possession Lens (Python3.12, GRU)"
python -m jupyterlab notebooks/01_Possession_Lens.ipynb
```

Choose the **Possession Lens (Python3.12, GRU)** kernel. The notebook reads saved reports and runs tests by default; downloads, training and challenge inference are disabled. [Full reproduction guide](docs/REPRODUCIBILITY.md).

## Model and evidence

Candidate053 combines an externally trained two-layer bidirectional GRU, tree-based control/dead-ball experts, a retained context RNN and structured H/A/D decoding. It uses508 observation features and offline future context. Missing ball is not automatically dead play; there is no hard minimum possession duration.

The mean official possession-spell F1 is **0.628641** on two heavily reused Bundesliga development matches across four tracking conditions. A frozen different-error-draw check on the same games scores0.622752. These are not independent-match tests, challenge accuracy or competition ranking. A separate partial-provider check regressed; transfer remains a limitation. [Technical document](<docs/submission/Visca el Barça, visca Catalunya.md>).

Later recovery/calibration experiments are preserved but failed release safeguards. The strongest bounded recovery prototype scores0.631917 on reused development data but adds false positives. Automatic Smart Tagging was audited; an external provider-event phase pilot lost genuine recoveries and does not validate transfer to Metrica tags. Neither replaces053. [Research reports](reports/external/) · [Release status](docs/submission/RELEASE_STATUS.md).

## Project layout

| Location | Purpose |
|---|---|
| `controlswitch/` | Inference, observation features, external research and website source |
| `notebooks/` | Current-first local notebook |
| `evaluation/` | Official CSV evaluation notebook and vendored evaluator |
| `models/` | Frozen submission assets; rejected experiments in `archive/` |
| `reports/` and `docs/` | Protocols, measured limitations and reproduction instructions |
| `examples/` and `dashboard.html` | Shareable named-match website examples |
| `data/`, `local/`, `work/` | Private inputs/caches/results; ignored by Git |
| `exports/` | Local submission bundle and source archive; ignored by Git |

All training and parameter tuning use external data. No provided-game labels/video are used for fitting or selection, no outputs are checked against its video, and no provided assets are uploaded to AI services. Each judge supplies their own licensed data. Source code and pinned weights are separate from match assets.
