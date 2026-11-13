# Project map

The canonical working copy is `Desktop/Possession Lens`; GitHub Desktop should point there. Open the root `dashboard.html` or Mac launcher for the judge experience. Keep those entry points stable.

| Area | Start here | Purpose |
| --- | --- | --- |
| Browser interface | `controlswitch/web/index.html`, `app.js`, `style.css` | Source for the single offline match centre |
| HTML generation/API | `controlswitch/dashboard.py`, `server.py` | Offline export and optional local server |
| CLI | `possession_lens/` | Project command entry point |
| Frozen inference | `controlswitch/rnn_runtime.py` | Validated checkpoint loading and label-free inference |
| Latest gated research | `controlswitch/gated_rnn.py`, `gated_experiment.py`, `gated_decoder.py` | Optional GRU/LSTM training and validation-only refinement |
| Other experiments | Other modules in `controlswitch/` | Retained for reproducible historical commands; not default entry points |
| Judge notebook | `notebooks/01_Possession_Lens.ipynb` | Saved reports, disabled-by-default reproduction switches |
| Match summaries | `examples/match-library.json`, `examples/matches/` | Shareable derived external examples, not private raw data |
| Weights | `models/README.md` | Production/gallery roles and archived research candidates |
| Evidence | `reports/README.md` | Current reports, protocols and preserved historical results |
| Documentation | `docs/UI_GUIDE.md`, `MODEL_CARD.md`, `RULES_BOUNDARY.md` | Interface, interpretation and competition rules |
| Provenance | `docs/data-sources/`, `docs/licences/`, `vendor/` | Data notices and upstream metrics |

Private input files remain under `data/`; generated run history remains under `local/research/`. Both are ignored. Duplicate Finder/iCloud copies are moved to `local/archive/finder-copies/` with their relative paths and an inventory; they are not deleted or uploaded. Backups before replacement live outside the project under `Documents/Codex/Backups`.

The root README is the current entry guide. `docs/RESEARCH_HISTORY.md` preserves earlier development notes; historical commands are run from the repository root. Modules/checkpoints referenced by those experiments retain their paths to avoid breaking reproduction.
