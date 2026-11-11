# Possession Lens

A clear view of team possession, built for More Than a Hack in Barcelona.

## Open the match centre

**On this Mac:** double-click **Open Possession Lens.command**. It opens the private local match analysis when available, otherwise the clearly labelled demo.

**On any computer:** open **dashboard.html** in a browser. No installation, server or internet is needed. This shared file contains invented demo data. Use **Open match analysis** to load a local analysis JSON or possession CSV; files are read in your browser and never uploaded.

The dashboard brings team percentages, control time, possession spells, a match timeline, anonymous track estimates and external model evidence into one interface. The palette is inspired by Barcelona; this is an independent hackathon project.

## Recreate an organiser-match analysis

Use Python 3.12. Obtain licensed match files independently; they are not on GitHub.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m possession_lens organise-data --source /path/to/organiser-downloads
python -m possession_lens rnn-predict \
  --data-root data/raw/provided --model models/rnn-boundary-001.npz \
  --out local/run
python -m possession_lens dashboard --run-dir local/run --out local
```

Then open `local/dashboard.html`, or use the Mac launcher. Inference refuses nonempty output run folders; use a new folder for a new run. The export step renders existing predictions and does not train, calibrate or tune anything.

For Jupyter, install `requirements-notebook.txt` and double-click **Open Notebook.command**, or run `python -m jupyterlab notebooks/01_Possession_Lens.ipynb`. Downloads, training and inference switches are off by default.

## Use files already in Downloads

On this Mac, double-click **Import Local Data.command**. It verifies and copies supported completed external XML files into `data/external/raw`, keeps the originals, ignores partial downloads and writes a local manifest. It never uses the network. Existing parsed match caches stay in `data/external/cache`.

```sh
python -m possession_lens import-external --source ~/Downloads
python -m possession_lens player-report --run-dir local/run --out local/player-control
```

The player report exports anonymous-track contributions, a frame-based control-interval ledger and a JSON summary. Its checks reconcile every track and unassigned interval against team control time. The dashboard's Player control tab also saves a CSV for the selected team. These are full-match proximity estimates, not verified individual-player possession.

The older `download-external` command now checks local Downloads first and requires `--allow-network` before fetching anything missing. Optional missing metadata does not block local import or existing cached team experiments.

## Folder guide

| Folder | What belongs here |
|---|---|
| `controlswitch/`, `possession_lens/` | Python engine and command entry point; dashboard source is in `controlswitch/web/` |
| `models/` | Frozen checkpoints; `rnn-boundary-001.npz` is recommended |
| `reports/external/` | External evaluation reports and model evidence |
| `docs/` | Model card, competition rules and research history |
| `notebooks/` | Reproducible Jupyter walkthrough |
| `data/` | Licensed local inputs and external caches; ignored by Git |
| `local/` | Private predictions, match JSON and private dashboard; ignored by Git |
| `exports/` | Shareable source packages; ignored except the folder guide |
| `tests/`, `vendor/` | Checks and the organiser's licensed reference code |

The Desktop folder is the canonical working repository connected to `hydrosmasher/possession-lens`. Keep only that copy in GitHub Desktop. Previous copies and historical private runs are preserved in the separate backup folder rather than mixed into the working project.

## What player control means today

The supplied slots are anonymous, fragmented tracklets, not stable player IDs or shirt numbers. The dashboard therefore shows **track-associated proximity estimates**, not named-player possession percentages. Missing evidence, other-team nearest tracks and numerical distance ties remain unassigned. Shares include unassigned time in the team-possession denominator. A per-track control-bout ledger is exported for future automatic identity work. These estimates do not confirm ball contact; nearby ball-flight frames can still be associated.

The first motion-based identity-linking experiment remains disabled: its best external validation setting had 3,570 correct links and 18 false links (99.4983% precision), below the predeclared 99.5% requirement. The held-out identity test was therefore not run. See `reports/external/identity-link-002-report.json`; no competition tracks were merged. Synthetic fragmentation is easier than real tracking errors.

Reliable individual-player statistics require automatic identity association and independent validation. Manual identity corrections on the provided match are prohibited. No club/player names are inferred from slot numbers.

## Model evidence and rules

The latest boundary-aware model modestly reduces clean missed spells from 246 to 237 and false-positive spells from 81 to 75 on an external Fortuna Düsseldorf–St. Pauli match. Clean spell F1 is 0.5025 versus baseline 0.3608 and previous-model 0.4860. Ball-dropout performance is essentially unchanged from the previous model. Clean merging remains unresolved. This match shares a club with training; no competition accuracy or statistical-significance claim is made.

See [the model card](docs/MODEL_CARD.md), [full recovery experiment](docs/research/RNN_RECOVERY.md), [rules](docs/RULES_BOUNDARY.md), and `reports/external/rnn-boundary-001-report.html`. All training/tuning uses external data. No competition truth/video checks occur. Inputs and generated organiser-match analyses stay local.

```sh
python -m unittest discover -s tests -q
node tests/test_dashboard.cjs
python -m possession_lens export-demo --out dashboard.html
```

**Authors:** Hrishikesh Karmalkar, Basel, Omar, Michail and Paul. [Source on GitHub](https://github.com/hydrosmasher/possession-lens).
