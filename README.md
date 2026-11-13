# Possession Lens

An offline match explorer for **More Than a Hack**, built by Hrishikesh, Basel, Omar, Michail and Paul. Explore team control, named-player involvement and verified match events in one Barcelona-inspired dashboard.

## Open the demo

**Open [dashboard.html](dashboard.html)** in your browser, or double-click **Open Possession Lens.command** on macOS. No installation, sign-in or server is needed for the bundled demo. Your files stay on your device.

1. Choose one of ten external matches. Barcelona–Real Madrid opens by default.
2. Select **Team control** and click a time window or possession segment.
3. Open **Player comparisons** to compare provider-mapped names. Unresolved control remains visible.
4. Select **Passes & shots** to open Bochum–Leverkusen, the example with verified events.
5. See **Model evidence** for frozen test results and the separate validation experiment.

Possession and player control are **model estimates**. Passes, shots, goals and substitutions appear only when a verified provider feed is available. Touch totals are not supplied. Player comparisons use full-match totals; half/window filters apply to the match view.

## Current model status

| Use | Checkpoint / result |
| --- | --- |
| Full-match production default | `models/rnn-boundary-001.npz` |
| Named external match gallery | Frozen context RNN; bundled examples are unchanged |
| Saved eligible research candidate | `models/archive/rnn-gated-decoder-004-validation-only.npz` |
| Saved candidate validation | Mean clean/dropout spell F1 **0.5554**, versus **0.5527** before decoder refinement |

The latest candidate is **validation-only** and has not replaced the default or gallery model. Reused partial provider labels do not establish fresh-match or challenge accuracy. The short-spell audit distinguishes label-gap fragments from actual rapid predicted switches. [Read the decoder study](docs/external/GATED_DECODER_004.md).

The unrestricted development peak is **0.5837 validation spell F1**, but that trial lost2 recoveries and increased merges. The new compact paired-switch comparison reaches **0.5832**, retains all27 recoveries, keeps merges66 and reduces false positives122→115. It still exceeds the strict112 false-positive guard; the saved model remains unchanged. [Latest studies and limits](docs/external/EVENT_DURATION_012.md).

## Run the notebook

Use Python3.11 or3.12 for a fresh setup:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
jupyter lab notebooks/01_Possession_Lens.ipynb
```

On macOS, **Open Notebook.command** can also use a locally installed Jupyter environment. The notebook reads saved reports by default; imports, retraining and challenge inference are disabled until explicitly selected. Optional GRU/LSTM training needs `requirements-gated.txt`. [Reproduction guide](docs/REPRODUCIBILITY.md) · [Gated model guide](docs/external/GATED_RNN_003.md).

## Find your files

| Folder | Contents |
| --- | --- |
| `controlswitch/` | Python pipeline, model implementations and browser UI source in `web/` |
| `possession_lens/` | Command-line entry point |
| `notebooks/` | Current reproducible Jupyter notebook |
| `examples/` | Ten shareable named-match summaries and verified event summary |
| `models/` | Frozen production/experimental weights; research candidates in `archive/` |
| `reports/` | Aggregate external evaluation evidence and experiment protocols |
| `docs/` | UI, data, model, competition and research documentation |
| `tests/` | Model, data-integrity and dashboard interaction checks |
| `vendor/` | Upstream challenge starter and metrics with their notices |
| `data/`, `local/` | Private local inputs and run history; excluded from GitHub |
| `exports/` | Local shareable source package; generated ZIP excluded from GitHub |

[Full project map](docs/FOLDER_GUIDE.md) · [Judge walkthrough](docs/UI_GUIDE.md) · [Research history](docs/RESEARCH_HISTORY.md) · [Model index](models/README.md) · [Report index](reports/README.md)

## Validate and share

```sh
python -m unittest discover -s tests -q
node tests/test_dashboard.cjs
node tests/test_gallery_ui.cjs
```

PyTorch-specific tests skip if the optional dependency is absent. Commit source, cleared notebooks, permitted model weights, licence notices and aggregate external reports. Provided-match data, video, ground truth and local predictions must remain private. Training and tuning use licensed external data only; predictions are never checked against the provided video. [Competition boundary](docs/RULES_BOUNDARY.md).

Blaugrana-inspired and independently built; no claim of official FC Barcelona affiliation.
