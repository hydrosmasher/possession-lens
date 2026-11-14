# Dead-ball reverse check015 and submission reproduction

The original014 candidate is retained. A second binary dead-state model was trained on J03WN1 and evaluated on J03WMX, reversing the two previously used Bundesliga matches. Decoder settings were frozen at turnover12, stop/restart3, dead log-odds weight0.75 before this run; no reverse-match hyperparameter selection occurred. Exactly one decoder configuration was evaluated. Same estimator, sample stride and augmentation recipe as014. Both halves stay within a match's role.

| Reverse check on J03WMX | Clean spell F1 | Ball-gap spell F1 | Mean |
| --- | ---: | ---: | ---: |
| Baseline smooth50 | 0.3072 | 0.2299 | 0.2686 |
| Production RNN | 0.4328 | 0.3742 | 0.4035 |
| Context RNN | 0.2559 | 0.1964 | 0.2262 |
| Factorised H/A/D | **0.5080** | **0.4395** | **0.4737** |

Candidate dead-state F1:0.9979 clean,0.9786 masked. Together with014, improvement appears in both directions. Neither match is fresh. The014 decoder was selected using WN1, which becomes training here, so this is a robustness check, not unbiased nested cross-validation. Both clubs and all historical data exposure remain relevant limitations. Provider-labelled states and complete original ball tracks can make external tasks easier than noisy challenge tracking. The reverse model is preserved locally and is not used for the challenge candidate.

Data: Bassek et al. (2025), IDSSE/Sportec, DOI10.1038/s41597-025-04505-y, CC BY4.0, first-division Bundesliga matches only for new dead-state training. Label audits, source/cache hashes, training sample counts and model hashes are in reports/external/deadball-015-reverse-*.json. No new downloads. See [014](DEADBALL_014.md) for feature and label semantics.

## Reproduce the reverse experiment

```sh
python -m pip install -r requirements-deadball.txt
python -m controlswitch.deadball --cache-root data/external/cache --out local/deadball-015-reverse-new --team-model models/rnn-europe-context-001-experimental.npz --production-model models/rnn-boundary-001.npz --reverse-check
```

Training and scoring use external caches only. `--reverse-check` loads the saved014 decoder settings and does not run the original eight-setting sweep.

## Generate the challenge candidate

Use Python3.11/3.12, install base requirements. No sklearn, PyTorch, internet, video or ground truth is needed for frozen candidate inference:

```sh
python -m pip install -r requirements.txt
python -m controlswitch.submission --data-root /path/to/provided-data --out local/submission-candidate-a
python -m controlswitch.submission --data-root /path/to/provided-data --out local/submission-candidate-b
python vendor/mtah/evaluation/validate_submission.py local/submission-candidate-a/possession.csv
```

The default command loads frozen014 settings, the saved context RNN, and the014 dead estimator. Both checkpoint hashes must match settings. The provided metadata and tactical tracking are read and strictly validated, never fitted. One-based ATD counters map to zero-based output frames. Coordinates become centred pitch metres. Inference resets per published half; outside the halves labels remain X. Separate output directories prevent overwriting previous results. Automatic tags and homography may be discovered but are not used; video is not read. No player identities are inferred.

Outputs stay private under ignored local directories: possession.csv, team-statistics.json, timeline.json and manifest.json. Model and actual input hashes are recorded. Possession percentages are predictions, not verified match statistics.

Two separate source directories and Python processes generated identical CSVs with199,319 rows and valid labels:

`cc5b4d6fb19cefa3b16a2f563a7a5b0dfa4dfdb3d3c18cd41959a3b272f50d91`

Both files passed the official vendored format validator. Full reproduction audit: reports/external/deadball-015-replay-check.json. This was the same installed dependency environment and hardware, not an independent OS/hardware check or clean dependency installation. No challenge score was calculated. Its predictions were not compared against video, public labels or other answer data, and were not used to choose the model.

## Submission status

This is a generated candidate, not a submitted entry or a production-model replacement. Final technical document, public repository/privacy audit, exact registered team filename and form upload still need completion. Preserve the previous local/run/possession.csv as a fallback. Do not publish match assets or reconstructive tracking exports. Training and all model selection remain external-only under the organisers' emailed clarification.
