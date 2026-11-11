# Possession Lens — team possession RNN

## Latest model: boundary-aware symmetric RNN

Recommended: `models/rnn-boundary-001.npz`. On fresh Fortuna Düsseldorf–St. Pauli tracking, clean spell F1 improves from 0.4860 to 0.5025, clean misses fall from 246 to 237 and false-positive spells from 81 to 75. Ball-dropout F1 is essentially unchanged (0.4232 to 0.4234). No sub-0.4-second predicted spells appear; clean merge-overlap count remains unchanged. The more aggressive spell-balanced model is experimental because it increases false positives and worsens dropout F1. See [RNN_RECOVERY.md](RNN_RECOVERY.md) for complete results, limits and reproduction.

## Previous model: team-symmetric RNN

The frozen team-symmetric checkpoint improves clean spell F1 on a new Bochum–Leverkusen test from 0.2884 (previous RNN) to 0.3707, versus baseline 0.2838. Under synthetic ball dropout it scores 0.2883 versus 0.2426 and 0.2083. The recurrent weights are unchanged; exact team-permutation averaging improves transfer. See [RNN_SYMMETRY.md](RNN_SYMMETRY.md) for the experiment history, limits and commands. Use `models/rnn-symmetric-001.npz`; this candidate has one independent external test.

## Reduced-switching RNN

The external-validation-selected offline decoder improves spell F1 from 0.3408 to 0.5138 on clean data and from 0.2787 to 0.4848 under ball dropout. This is validation evidence from one match, not a held-out-match result. See [RNN_IMPROVEMENT.md](RNN_IMPROVEMENT.md) for the full comparison and three-match reproduction command. Use `models/rnn-decoded-001.npz` to apply the frozen decoder.

## Open in Jupyter

On this Mac, double-click **Open Possession Lens.command** in the project folder. It uses the existing local Jupyter environment when available and opens `notebooks/01_Possession_Lens.ipynb` with a named project kernel. The existing local Python 3.15 / NumPy 2.5.4 / pandas 3.1.1 environment passed the project's 19 tests; the canonical pinned reproduction environment remains Python 3.12 with requirements.txt.

For a fresh judge checkout, use Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --sys-prefix --name possession-lens --display-name "Python 3 (Possession Lens)"
python -m jupyterlab notebooks/01_Possession_Lens.ipynb
```

The notebook's downloads, training and competition inference switches start disabled. Its default cells organise local links, inspect the frozen external validation report and run unit tests. Notebook outputs are cleared for sharing. The private development repository is https://github.com/hydrosmasher/possession-lens.

**Team:** Hrishikesh Karmalkar, Basel, Omar, Michail and Paul. See AUTHORS.md and PROJECT_PLAN.md.

The group's current direction is a recurrent neural network for frame-level team possession, followed by team-possession percentages. Earlier classical Possession Lens models remain in this repository as comparators. Initial individual outputs are unverified **tracklet proximity proxies**, not named-player contribution statistics.

## RNN workflow for judges

Use Python 3.11–3.13 and the pinned requirements below. No PyTorch, GPU, or paid service is needed. This is an actual tanh recurrent network with 24 hidden units, softmax H/A/D output and truncated backpropagation through time. Hidden state is carried across successive chunks and reset at period boundaries. Training and inference use 25 Hz observations. This small first model is not yet an optimised submission.

```sh
python -m possession_lens organise-data --source "/path/to/organiser-downloads"
python -m possession_lens download-external --out work/external/raw
python -m possession_lens rnn-train --external-root work/external/raw \
  --out work/rnn/train-001 --epochs 5
python -m possession_lens rnn-predict --data-root data/raw/provided \
  --model work/rnn/train-001/rnn-model.npz --out work/rnn/predictions-001
```

The organiser's downloads are linked, not moved or duplicated. The default layout is:

```text
data/raw/provided/
  tactical/   # video, tracking XML/TXT, homography CSV, Smart Tagging XML
  broadcast/  # optional separate timeline, not used by initial RNN
work/external/raw/    # licensed external training files
work/external/cache/  # parsed external data, private and ignored
work/rnn/             # training runs and local predictions, ignored
models/               # small frozen external-trained checkpoints suitable for sharing
outputs/              # shareable synthetic website, documentation and source package
```

For inference without retraining, use the supplied `models/rnn-pilot-001.npz` checkpoint with the `rnn-predict` command. Its pilot training protocol is in `models/rnn-pilot-001-report.json`. Training uses only external Sportec matches. The initial local run uses an existing external cache and `--single-match-pilot`; it is validation-only development evidence and does not establish cross-match performance. The three-match setup uses separate J03WOH training and J03WPY validation matches; this RNN starter deliberately does not score the previously viewed test again.

Prediction outputs:

- `possession.csv`: all 199,319 tactical video frames, with X outside the published graded blocks.
- `team-statistics.json`: H/(H+A) and A/(H+A), excluding dead and unknown frames; also their durations and label coverage over the whole video.
- `tracklet-proxies.json`: nearest-track geometric association during predicted team possession, with missing/in-flight time explicitly unassigned. This is **not** learned player attribution.
- `timeline.json`: local predicted label intervals.

The RNN reads tactical player/ball positions and speeds. Homography, tags and videos are organised but not used by this first recurrent model. There is no training or tuning on provided-match ground truth or video and no output-versus-video check.

## GitHub preparation

Source, requirements, authors, model card, small external-trained model checkpoints and the vendored licence can be committed. `data/`, `work/`, original videos, external raw inputs and local generated predictions must stay excluded. Each judge obtains licensed match assets independently. Do not commit local symlink manifests or absolute machine paths. The shareable source is published at https://github.com/hydrosmasher/possession-lens.

The earlier workflow and experiment details follow.

A Python possession-research pipeline and an offline-friendly local website for judges.

**No-install website:** open `outputs/possession-lens-demo.html` directly in a browser. It is a self-contained synthetic demo with CSV inspection. It requires neither Python nor a server. Regenerate it with `python -m possession_lens export-demo` after installing the dependencies below.

**Implemented:** strict tactical tracking ingestion, reproduction of the four official nearest-player baselines, official CSV format validation, run provenance, interactive timeline comparison, and browser-local CSV inspection.

**Model milestone:** an externally trained three-state classifier, a learned boundary verifier and structured decoding are implemented. A separate development experiment adds 2D interaction evidence and strictly trailing-feature one-second turnover forecasting. Read the external experiment reports for measured results and limitations; none is a competition score.

## Quick start: judge walkthrough

Use Python 3.11–3.13 (3.12 recommended), then from this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m possession_lens serve
```

Open **http://127.0.0.1:8765**. On Windows activate with `.venv\Scripts\activate`.
The default demo is deterministic and entirely synthetic. It uses no licensed match assets and does not simulate a trained model. The server binds only to loopback. The frontend has no package manager, compilation, CDN, analytics, remote fonts, or external requests. Dependency installation initially needs internet; running the installed project does not.

## Reproduce a baseline with the licensed dataset

Obtain the match assets from the organisers separately. Do not include them in this project. Both the original tactical folder structure and the split download folders work; discovery is recursive and rejects duplicate inputs.

```sh
python -m possession_lens baseline \
  --data-root "/path/to/MORE THAN A HACK" \
  --out work/runs/baseline-001

python -m possession_lens serve --run-dir work/runs/baseline-001
```

Use a new, empty output folder per experiment. The pipeline refuses to overwrite existing runs. There is no training step. It reads the tactical tracking metadata and TXT, not the video. Tagging and homography are discovered and recorded as optional assets but are **not used by the baseline**.

Outputs in the ignored run folder:

- `baseline_raw.csv`, `baseline_smooth25.csv`, `baseline_smooth50.csv`, `baseline_smooth125.csv`: full-length `frame,label` submissions.
- `summary.json`: descriptive results for the published graded halves, runtime versions, output hashes, and validation status.
- `timeline.json`: local predicted label intervals for inspection.
- `manifest.json`: local input paths, sizes, and hashes of inputs actually used.

All generated match results are local research artifacts. Do not publish these folders or serve them on public hosting. Input paths are not exposed by the website. The CSV inspection picker parses files in the browser and never uploads them.

To use a free port: `python -m possession_lens serve --port 8766`.

## Method and alignment

- Official reference: `metrica-sports/more-than-a-hack-2027`. The upstream commit is recorded under `vendor/mtah/UPSTREAM_COMMIT`.
- The supplied ATD counter is one-based: submission frame = ATD frame ID minus one. IDs must be consecutive and there must be exactly 199,319 records.
- Positions are normalised. Nearest-player distance uses 105.3 × 68.0 metres, matching the official baseline.
- Team metadata order follows the organiser's baseline: first team is home. Slots must belong to one of the two declared teams.
- The raw baseline marks unavailable evidence as `D`. This is a baseline limitation, **not a claim that a missing ball means stopped play**.
- Smoothing uses the upstream implementation unchanged. Windows are centred at ±25, ±50 and ±125 frames, not 25/50/125-frame total windows. All video frames are retained, including ungraded periods, to reproduce the official algorithm.
- Descriptive possession counts in the website are restricted to each graded half independently. Smoothing itself follows the official full-array implementation. These are prediction statistics, never ground-truth match statistics.
- Homography maps normalised pitch coordinates to image pixels. The baseline already has pitch coordinates, so it does not need image projection.

## Validation and reproduction

```sh
python -m unittest discover -s tests -v
python vendor/mtah/evaluation/validate_submission.py \
  work/runs/baseline-001/baseline_smooth50.csv
```

Optional developer check, with Node.js installed: `node tests/test_web.cjs`. This uses a DOM test double for interactions and does not substitute for visual browser QA.

Re-run into a different folder and compare the CSV SHA-256 hashes in the summaries. The output CSVs should be identical. Summary runtime versions may differ across environments; the dependencies here are pinned to the versions used for the first run.

The organiser's latest emailed clarification takes precedence over older repository guidance. Do not check predictions against the competition video. Do not train, fit, calibrate or tune on the competition possession ground truth or video, including the public possession labels. All scoring, training and model selection in the new experiment commands use external data only. Automatic format and consistency checks on competition inputs are allowed. Competition videos and data extracts must not be sent to AI services. See RULES_BOUNDARY.md.

## Next development milestone

Acquire fresh external matches and evaluate cross-match transfer before promoting the detector. The original pilot uses a separated first-half development split and an unseen second-half test of one match; this is weaker than a match-held-out benchmark. Later interaction and forecasting results use validation only because the original test has already been viewed. The current model does not yet use Smart Tagging or video.

## External experiments

No ML framework is required: the models use NumPy and the existing official scorer. Download pinned external IDSSE/Sportec data (CC BY 4.0) separately:

```sh
python -m possession_lens download-external --out work/external/raw
python -m possession_lens benchmark --external-root work/external/raw --out work/experiments/match-pilot-001
```

The three-match pilot fixes J03WOH for training, J03WPY for validation and J03WMX for test. The downloader requests only match information and positions, not video. The current pilot was run with the one complete downloaded match available:

```sh
python -m possession_lens benchmark --external-root /path/to/downloads \
  --out work/experiments/pilot-001 --single-match-pilot
```

It splits the first half at 70%, purges 100 frames on each side, trains before the gap, selects parameters after it, and tests once on the whole second half. Features are computed separately within each partition. Ball gaps are independent of reference labels. No random frame train/test split is used.

```sh
python -m possession_lens develop --external-root /path/to/downloads \
  --pilot-dir work/experiments/pilot-001 --out work/experiments/development-001
```

This trains interaction-enhanced detection and causal forecasting on the original training split and reports validation performance only. It does not score the previously viewed test half again. Interaction candidates are not confirmed contacts. The forecasting target is a direct H↔A change within one second before a stoppage; dead/unknown current frames and incomplete horizons are excluded. Forecast features contain no future observations and no offline-detector output; source tracking itself may have been smoothed offline. Reported forecast metrics are frame-level, not event-level alarm performance.

`model.npz` and `development-models.npz` store weights, standardisation statistics and feature-version metadata. Experiment folders also contain frozen protocols, selection records, external-only intermediates and HTML/JSON results. Reuse reports honestly: seeing a test result makes later model changes exploratory on that same test.

## Attribution

Official baseline, constants, metrics, and validator are vendored unmodified from [Metrica Sports' challenge repository](https://github.com/metrica-sports/more-than-a-hack-2027), under its accompanying licence. Match assets have separate competition-only terms. This project's original source is MIT licensed; the MIT licence does not grant rights to any match data.
