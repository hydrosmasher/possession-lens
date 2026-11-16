# Possession Lens

Local H/A/D possession segmentation for **More Than a Hack**, with an offline match dashboard. Built by Hrishikesh, Basel, Omar, Michail and Paul. Registered team: **Visca el Barça, visca Catalunya**.

## Reproduce frozen submission candidate 053

Use Python 3.12 and obtain the licensed match files separately:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gru-inference-lock.txt
python -m controlswitch.recurrent_pipeline --data-root /path/to/provided-data --out local/submission-053 --config reports/external/short-spell-053-config.json --filename "Visca el Barça, visca Catalunya.csv"
python evaluation/official/evaluation/validate_submission.py "local/submission-053/Visca el Barça, visca Catalunya.csv"
```

Starting benchmark: candidate 052 **0.615370**. Final checkpoint: **053, overall external development possession-spell F 1 0.628641**. The target is 0.85–0.90; it remains unmet, and competition rank is unknown.

The selected GRU continues 052 for 12 fixed iterations × 512 windows (6,144 additional sampled windows; 52 iterations and 26,624 sampled windows including its 043/052 history). Half of its training anchors are drawn uniformly from known-live spells of at most 3 seconds, then uniformly within that spell. Those frames receive twice their previous spell-weighted CE weight. AdamW learning rate 0.0001, weight decay 0.001, gradient clipping 1, seed 153, dropout 0.25, spatial reflections and team swaps remain fixed. The unchanged two-layer bidirectional 64-unit GRU still uses 508 features, projection 64, LayerNorm and the narrow auxiliary boundary objective. Forward weights fit WMX only; reverse development fits WN 1. Labels affect training sampling/loss only and never enter inference.

The two Bundesliga matches are heavily reused external development data. Four conditions pool opposite-match predictions: clean 0.727686, short ball gaps 0.636442, long gaps 0.610947, tracking noise 0.539490. These use the organisers' official spell metric, not the challenge ground truth. No challenge video, labels or public score is used for fitting, selection or comparison.

After the fixed winner was selected, different simulated-error draws scored 0.606253 → 0.622752. These are the same matches, not an independent-match test.

Across repeated conditions, misses 1746 → 1603, false positives 915 → 1101, and geometric merges 422 → 369. These counts repeat physical possessions across conditions. False detections and missed possessions are both reported; a higher F 1 does not imply every error category improved.

The current-first notebook runs with the **Possession Lens (Python 3.12, GRU)** kernel. Optional training/inference stay disabled by default. For a new notebook environment, run `python -m pip install -r requirements-gru-notebook.txt`, then `python -m ipykernel install --user --name possession-lens-gru-312 --display-name "Possession Lens (Python 3.12, GRU)"`. Select that kernel when opening the notebook. CPU inference needs no GPU or hosted service. Full CSV reproduction, frozen asset hashes and external replays are saved in reports/external/. The gallery retains its separate frozen model; current submission evidence does not imply gallery examples were recomputed. [Session evidence](docs/submission/DAY_2026_11_15.md).

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
| Frozen Phase 1 candidate 053 | `reports/external/short-spell-053-config.json`, pinned recurrent/base assets |
| Legacy full-match default | `models/rnn-boundary-001.npz` |
| Named external match gallery | Frozen context RNN; bundled examples are unchanged |
| Saved eligible research candidate | `models/archive/rnn-gated-decoder-004-validation-only.npz` |
| Saved candidate validation | Mean clean/dropout spell F 1 **0.5554**, versus **0.5527** before decoder refinement |

## Preserved earlier research

The saved gated candidate is **validation-only** and has not replaced the default or gallery model. Reused partial provider labels do not establish fresh-match or challenge accuracy. The short-spell audit distinguishes label-gap fragments from actual rapid predicted switches. [Read the decoder study](docs/external/GATED_DECODER_004.md).

The unrestricted development peak is **0.5837 validation spell F 1**, but that trial lost 2 recoveries and increased merges. The new compact paired-switch comparison reaches **0.5832**, retains all 27 recoveries, keeps merges 66 and reduces false positives 122→115. It still exceeds the strict 112 false-positive guard; the saved model remains unchanged. [Latest studies and limits](docs/external/EVENT_DURATION_012.md).

The complete H/A/D development candidate now scores **0.4471 mean spell F 1** versus **0.2525 baseline** and **0.3537 production RNN**, on one reused Bundesliga validation match across clean and simulated ball gaps. Dead-state F 1 is 0.9977/0.9813. This is not a fresh test or challenge score; production remains frozen. [Dead-ball study](docs/external/DEADBALL_014.md).

The frozen decoder also improves the reverse Bundesliga check: **0.4737 mean spell F 1**, versus 0.4035 production and 0.2686 baseline. A new `python -m controlswitch.submission` command generates the complete challenge candidate; two separate source copies produced identical CSV hashes and passed format validation. No challenge accuracy was measured. [Reverse check and CSV reproduction](docs/external/DEADBALL_015.md).

The new hybrid control correction improves pooled full H/A/D development spell F 1 **0.4455→0.5024** across clean, short-gap and long-gap tracking in both reused Bundesliga directions. Misses 1939→1772, geometric merges 424→408 and false positives 716→569; some existing matches are lost and reverse-fold merges rise slightly. It remains a separate validation candidate. [Missing-ball studies 016/017](docs/external/CONTROL_017.md).

## Official evaluation notebook

Open [evaluation/01_Official_Evaluation.ipynb](evaluation/01_Official_Evaluation.ipynb). Save a frozen possession CSV beside it: a single CSV is selected automatically, or enter its name in `CSV_FILENAME`. It runs the organisers’ unchanged public evaluator offline; results stay in ignored local folders. The five-minute reference is for pipeline checks, never model tuning.

## Run the notebook

Use Python 3.11 or 3.12 for a fresh setup:

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

## Separate provider check: a regression to investigate

On the previously reused PSG–Marseille Ligue 1 example, mean partial-label spell F 1 declines **0.300976 → 0.278404** for 052→053. This is a different-provider diagnostic, not a fresh test. Its censored team-possession groups have no verified live/dead labels, so these values cannot be equated to the complete H/A/D benchmark. It was frozen after selection and was not used to refit, tune or reselect the model. The regression exposes transfer risk:053 is a stronger candidate on the current Bundesliga benchmark, not a demonstrated universal improvement.052 remains preserved as a fallback. See reports/external/provider-056-report.json.
