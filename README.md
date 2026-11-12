# Possession Lens

A clear view of team possession, built for More Than a Hack in Barcelona.

## Ten named matches, one offline website

Open `dashboard.html` or double-click **Open Possession Lens.command**, then use **Choose your match**. The library contains nine historical SkillCorner examples plus Bochum–Leverkusen. Team possession, timelines, named-player estimated impacts, search and exports update with the selected match. The Model evidence tab now shows the frozen context-RNN result (0.6286 vs 0.5804 previous European RNN), not the older capacity experiment.

Every displayed person is joined to a provider roster. Anonymous tracks and missing evidence stay unresolved; no guessed names or anonymous player rows are shown. Mapped proximity control covers about 34–59% of team control across these examples, with the rest explicitly unresolved. This is estimated involvement, not confirmed ball contact, passing impact or causal value. The H/A-only examples show stopped play as “Not classified”. Challenge individual rows remain hidden because no permitted name mapping exists.

A finer decoder search on development data did not pass the improvement criteria; the context checkpoint remains unchanged. No further anonymously open, properly licensed top-five tracking + possession dataset was verified, so there is **no new fresh-match score**. Free-but-gated Premier League collections are documented for possible future access; event-only data are not substituted for tracking truth. [Source audit](docs/data-sources/OPEN_ACCESS_AUDIT_2026-11-12.md). The latest notebook includes the gallery, mapping coverage and this research status.

## Open the match centre

**On this Mac:** double-click **Open Possession Lens.command**. It opens the named external-match demo. Use Open local analysis to load your private match JSON when needed.

**On any computer:** open **dashboard.html** in a browser. No installation, server or internet is needed. This shared file starts with the named Bochum–Leverkusen external example. Use **Open match analysis** to load a local analysis JSON or possession CSV; files are read in your browser and never uploaded.

The dashboard brings team percentages, control time, possession spells, a match timeline, named external player estimates and model evidence into one interface. The palette is inspired by Barcelona; this is an independent hackathon project.

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

Then open `local/dashboard.html` or load `local/match-analysis.json` using Open local analysis. The Mac launcher keeps the named external demo as its default. Inference refuses nonempty output run folders; use a new folder for a new run. The export step renders existing predictions and does not train, calibrate or tune anything.

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

## Latest expanded-data temporal-context experiment

Training expanded from three to five top-five-league matches by adding Inter–Juventus and Liverpool–Manchester City. A data-only RNN and an RNN with surrounding tracking context were trained from scratch. The validation-selected **context** candidate was frozen before the fresh **Barcelona–Real Madrid (18 December 2019)** test. Neither earlier test was used for training or tuning.

Mean clean/dropout spell F1 on this same fresh match: **selected candidate 0.6286**, **previous European RNN 0.5804**, **baseline 0.4484**. Candidate beats previous: **True**. Clean missed spells: **125 → 115**; possible merge overlaps: **29 → 27**; false positives: **49 → 37**. Check dropout diagnostics too; a higher F1 can have tradeoffs.

Experimental recommendation: candidate. Production default unchanged: these partial H/A labels cannot validate full-match dead-ball behaviour. These are partial automatic H/A labels, not independently annotated full-match ground truth or official challenge accuracy. The target of 0.8–0.9 is not established by this experiment. The new experimental weights are `models/rnn-europe-context-001-experimental.npz`; validation-selected family ablations are archived separately. The notebook now shows this latest experiment before historical results, including optional verified import and reproduction cells. The test is now viewed; no post-test tuning was performed.

[Full results](reports/external/rnn-europe-context-001-report.html) · [Method, provenance and reproduction](docs/external/EUROPE_CONTEXT.md).

## Latest spell-recovery attempt: no new improvement

The latest notebook includes the recovery experiment and the new reserved **Bayern–Dortmund (9 November 2019)** test. Spell-balanced RNN loss and balanced boundary heads failed to improve validation within the false-positive limits, so they were rejected. The selected checkpoint is **identical to `rnn-europe-001-experimental.npz`**. Do not present the new match score as an algorithm improvement.

On the fresh match, mean clean/dropout possession-spell F1 was **retained European RNN 0.5625**, **baseline 0.3852**, **older production model 0.2494**. The retained European RNN's clean predictions still miss 151 labelled spells and have 37 possible merge overlaps. The 0.8–0.9 target remains unmet. Labels are partial and automatic, with no supervised dead-ball state; this is conditional transfer evidence, not challenge accuracy. Both clubs also appear in training. The match is now viewed and excluded from future tuning.

The notebook shows misses, false positives, merge overlaps, short pulses, loss history and the exact freeze decision. Rejected weights are preserved in `models/archive/rnn-europe-recovery-001-weighted-rejected.npz`; production defaults remain unchanged. [Full report](reports/external/rnn-europe-recovery-001-report.html) · [Recovery method and reproduction](docs/external/EUROPE_RECOVERY.md).

## Latest European transfer test and notebook

The updated `notebooks/01_Possession_Lens.ipynb` walks through the latest five-match experiment, source verification, split, label audit, training history and results. Double-click **Open Notebook.command**, then run cells from top to bottom. Training and challenge inference remain disabled by default.

A new RNN was trained from scratch on Manchester City–Liverpool, Dortmund–Bayern and Juventus–Inter, selected on PSG–Marseille, then tested once on reserved Real Madrid–Barcelona. Mean clean/dropout possession-spell F1: **candidate 0.5428**, **current 0.3088**, **baseline 0.4231**. The candidate wins this conditional transfer test; it has not reached 0.8–0.9. One layer at iteration 5 beat deeper and later checkpoints. The candidate produced 179 clean spells against 295 labelled reference spells, leaving under-segmentation unresolved.

These are partial automatic SkillCorner labels, with unknown gaps and no supervised dead-ball state. Scores are agreement on labelled runs, not complete-match independent ground truth or official challenge accuracy. The new weights are saved as `models/rnn-europe-001-experimental.npz`; **the production default remains unchanged**. Test labels were opened after freezing settings; this match is now viewed and must not be used for tuning. [Results](reports/external/rnn-europe-001-report.html) · [Protocol and reproduction](docs/external/EUROPE_EXPERIMENT.md).

## Fresh-match decision: keep the current model

A frozen test on previously unused Fortuna Düsseldorf–Kaiserslautern (J03WR9) favoured the current boundary-aware RNN. Mean clean/ball-dropout spell F1: **current 0.4913**, longer-trained candidate **0.4767**, baseline **0.2721**. Candidate clean/dropout scores were 0.5217/0.4316; current scores were 0.5322/0.4504. The longer-trained model is not promoted. Both had zero predicted spells shorter than 0.4 seconds; under dropout the candidate missed fewer spells but added more false positives.

The match and settings were declared before downloading/scoring; source and frozen weights were verified. It shares a club with training, and one new match is limited evidence. No parameters were tuned after seeing it. J03WR9 is now viewed; J03WOY remains unused. See [the full fresh-match report](reports/external/rnn-fresh-J03WR9-001-report.html).

The refreshed offline UI uses a dark blaugrana theme, larger possession figures, simpler Match / Players / Model evidence tabs, searchable player contributions and the fresh comparison. [UI guide](docs/UI_GUIDE.md). Visual browser QA remains unverified because local-file preview was blocked; automated checks passed.

## Named players and deeper RNN experiments

The dashboard opens the named external Bochum–Leverkusen match by default. There is no anonymous demo. Its player names and shirts are automatically joined by provider PersonId + team to the external roster. Shares remain estimated proximity associations. This example does not name players in the challenge match: the provided metadata contains generic names and sequential slot numbers, and no reliable mapping has been supplied. Never replace those slots with guessed squad names or manual video annotations.

We compared 1, 2 and 3 recurrent layers, each with 24 hidden units, for up to 32 sampled training iterations. Dropout, L2 regularisation, clipped gradients, fixed clean training/validation loss probes and early stopping are implemented. The one-layer model still gives the best validation spell segmentation; adding layers did not improve that objective. The longer-trained candidate is experimental, not the default. See [the learning curves and results](reports/external/rnn-capacity-002-report.html) and [reproduction instructions](docs/RNN_CAPACITY.md).

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
