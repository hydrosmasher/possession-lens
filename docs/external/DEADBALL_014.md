# Full H/A/D experiment 014: dead-ball detection

A factorised pipeline combines the frozen top-five-league context RNN's conditional H/A probabilities with a new binary live/dead gradient-boosted classifier. Three-state offline Viterbi uses separate turnover and stop/restart penalties. This addresses a previously unvalidated state in the recent context-RNN research.

| Model | Clean spell F1 | Ball-gap spell F1 | Mean spell F1 | Clean dead F1 | Ball-gap dead F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nearest-player baseline smooth50 | 0.2838 | 0.2213 | 0.2525 | 0.0000 | 0.0986 |
| Frozen production RNN | 0.3829 | 0.3245 | 0.3537 | 0.9882 | 0.9347 |
| Frozen context RNN | 0.2062 | 0.1382 | 0.1722 | 0.0094 | 0.0054 |
| Factorised H/A/D candidate | **0.5028** | **0.3914** | **0.4471** | **0.9977** | **0.9813** |

Mean gain: +0.1946 over the fixed baseline and +0.0935 over production. These are development measurements on reused validation, not independent test evidence or challenge accuracy. Do not compare them with historical partial-label SkillCorner scores or the official full-match baseline score.

## Sources and split

Existing IDSSE/Sportec CC BY 4.0 data, Bassek et al. (2025), DOI10.1038/s41597-025-04505-y. DFL competition000001 is Bundesliga, the German first division. Dead estimator training: J03WMX; validation: J03WN1 (Bochum–Leverkusen). Both halves remain together by match. The previous held-out status of WMX is retired: it was already viewed historically and is now explicitly training data. WN1 is reused validation, not fresh. No other matches, challenge inputs, video, public ground truth, or external human annotations of the challenge game are accessed.

Labels are provider BallStatus 0=dead and1=live, then BallPossession1/2 for H/A during live play. Unknowns are excluded. Both matches have full H/A/D coverage and complete original ball positions. Label/cache/source hashes and frame counts are in the audit. Their ball visibility is much cleaner than the challenge: simulated 40% contiguous missing-ball augmentation and stress evaluation are useful but cannot establish robustness to the real 41% missing ball, identity switches or geometric errors.

The frozen team backbone was trained separately on the documented SkillCorner top-five-first-division training pool. It is not retrained here. The production comparator has historical second-division training ancestry, which is disclosed rather than relabelled as first-division training.

## Model and protocol

New dead-state features are the mean and absolute difference of the existing120 geometry/context features under both team orders (240 total). They include ball visibility, kinematics, pitch-boundary distance, nearest-player geometry and collective movement. Feature functions never read labels or provider BallStatus/BallPossession. Missing ball is masked input evidence, not a prescribed D output. Context uses future observations within each half: this is offline inference.

Histogram gradient boosting:100 iterations,15 leaves, learning rate0.08, minimum40 samples/leaf, L2=10, no random validation or early stopping, seed114. Known training frames sampled every fifth frame, separately clean and masked, for58,388 training examples. Parameter grid fixed before fitting: team penalty12/24, stop/restart penalty3/12, dead log-odds weight0.75/1.0. Eight validation candidates; first maximum of mean clean/masked official spell F1 wins. Selected12/3/0.75. No further sweep after results.

Dead probability d yields [(1-d)*P(H|live), (1-d)*P(A|live), d]. Viterbi can represent H→D→H as two spells. The model has no hard minimum spell duration. This experiment does not yet prove the human concept of controlled possession is captured across domains.

No Smart Tagging is used. It remains a possible soft evidence source, but there is no audited matched external tag validation here. No manual match-specific constants or corrections were introduced.

## Reproduce

Python3.11/3.12, from project root:

```sh
python -m pip install -r requirements-deadball.txt
python -m controlswitch.deadball --cache-root data/external/cache --out local/deadball-014-new --team-model models/rnn-europe-context-001-experimental.npz --production-model models/rnn-boundary-001.npz
```

The two existing caches are created from licensed external XML by `controlswitch.external.load_sportec`. New run directory must not already exist. Exact cache and source hashes are published in `reports/external/deadball-014-label-audit.json`; obtain original datasets separately. Private frame arrays stay in ignored local folders.

For inference without sklearn, load the saved binary NPZ with `tree_model.load`, the frozen team backbone with `rnn_runtime.load_frozen`, then use `deadball.infer_candidate` with the published settings. The NPZ stores explicit numeric tree arrays, no pickle. Native sklearn/export probabilities agree on training probes and every validation frame. Settings include both checkpoint hashes. `load_candidate(run_dir, team_checkpoint)` validates local run hashes automatically. To load the published candidate:

```python
from controlswitch.deadball import load_candidate, infer_candidate
candidate = load_candidate(
    "reports/external/deadball-014-settings.json",
    "models/rnn-europe-context-001-experimental.npz",
    "models/archive/deadball-014-validation-only.npz",
)
labels, probabilities = infer_candidate(candidate, sequence, pitch)
```

`sequence` contains only observation arrays: xy, speed, teams, ball, ball_speed; pitch is length/width in metres. Calls reset at each half. For raw XML to cache, use the existing external loader:

```python
from pathlib import Path
from controlswitch.external import load_sportec
for mid in ("J03WMX", "J03WN1"):
    raw = next(Path("data/external/raw").glob(f"*positions_raw_observed*{mid}.xml"))
    load_sportec(raw, mid, Path("data/external/cache"))
```

All previous checkpoints, gallery examples and challenge prediction files are preserved. This candidate is saved under models/archive and has not replaced the production model. The notebook reports these results by default; retraining is opt-in. A full challenge CSV and byte-identical clean-checkout verification remain separate submission work.

References: [dataset paper](https://doi.org/10.1038/s41597-025-04505-y), [pinned sklearn estimator](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
