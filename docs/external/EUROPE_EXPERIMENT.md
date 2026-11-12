# European first-division possession experiment

This experiment uses the five requested matches from the official historical SkillCorner open-data release, pinned at commit `1ed50b775f93075c2a6d00d0d795f13b7b89709e`. The files are MIT licensed; see `docs/licences/SkillCorner-MIT.txt`. Inputs remain local and ignored by Git.

Split fixed before tracking inspection:

| Role | Match |
|---|---|
| Training | 4039 Manchester City–Liverpool, 3749 Dortmund–Bayern, 3518 Juventus–Inter |
| Validation | 2269 PSG–Marseille |
| Test | 3442 Real Madrid–Barcelona |

The downloaded ZIP contains the full historical repository, but only the ten metadata/tracking files for these five matches are imported. Their file sizes and Git blob hashes must match the pinned catalogue. Test contents are downloaded and checksummed without parsing, then loaded only after candidate and decoder settings have been saved in `frozen.json`.

## What the labels support

These are automatic provider possession labels, not independently annotated ground truth. Home and away are supervised; null is unknown (X), never dead ball. Only periods 1 and 2 are used; pre-match and half-time frames are excluded. Unknown intervals stay in the timeline and split graded runs. No labels enter model features. No competition files are accessed by this experiment.

During development, labelled native playing-frame coverage ranged from 61.2% to 76.2% per half. Median continuous labelled runs were 38.1–48.7 seconds, with 90–122 direct labelled team changes per half. This supports a conditional transfer experiment, but not evaluation of complete possession spells or dead-ball recognition. Unknown gaps can truncate spells and inflate scores. Compare models only on the identical label mask, and do not interpret these values as official challenge accuracy or evidence that the 0.8–0.9 target is reached.

Stable provider identities use fixed roster columns. Anonymous team tracks use a bounded separate pool; a newly assigned slot starts missing to avoid fabricating a movement vector. Referees are excluded. Tracking is interpolated between adjacent finite 10 Hz observations onto a common 25 Hz interface required by the existing checkpoint. Missing samples are not bridged. Labels use sample-and-hold, preserving unknowns. Interpolated frames are not additional independent evidence. Provider frame-clock spacing was checked against the supplied time field in all four development matches and agreed at 10 Hz.

## Training and selection

The new candidate is trained from scratch on the three top-five-league training matches, without second-division pretrained weights. One- and two-layer tanh RNNs have 24 hidden units per layer. Each has up to 12 sampled iterations, up to 1,024 windows per iteration, 128-frame windows, 16-window batches, dropout 0.15, L2 0.0001, Adam and gradient norm clipping at 5. A sampled iteration is not a full epoch. Team swapping and simulated contiguous ball dropout are used for augmentation.

The checkpoint and decoder are selected by mean clean/dropout validation possession-spell F1, over decoder penalties 3, 12, 24 and 48. Training/validation cross-entropy is diagnostic; later iterations are not automatically preferred. The baseline smoothing window is selected on the same validation match. The existing boundary-aware model is frozen as a comparator; its historical second-division ancestry is disclosed rather than included in the new training pool.

The selected model and settings are hashed and frozen before test loading. The reserved match is scored once with clean tracking and fixed simulated dropout (seed 300). No settings may be changed using test results. Re-running the same command later is reproduction on a viewed test, not a new independent test. The current production model is not automatically replaced: partial provider-label evidence alone does not establish a full challenge improvement.

## Reproduce locally

Install `requirements-notebook.txt`, open `notebooks/01_Possession_Lens.ipynb`, and run the default walkthrough. Training is disabled by default. The notebook contains verified archive import and optional training switches. Raw assets are not on GitHub.

Alternatively, after importing the five matches:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m controlswitch.skillcorner_europe \
  --root data/external/skillcorner-europe/raw \
  --reference models/rnn-boundary-001.npz \
  --out local/europe-reproduction-001 --iterations 12
```

Use a new output directory. A frozen experiment refuses refitting. Aggregate reports and the experimental checkpoint are shareable; private challenge data and raw match assets are excluded from the source package.
