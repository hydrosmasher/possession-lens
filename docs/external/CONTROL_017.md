# Missing-ball robustness016 and hybrid team-control017

The selected validation candidate combines the frozen context RNN with a new external-trained home/away correction and the existing014 dead-ball estimator. It improves complete H/A/D spell segmentation in clean, short-gap and long-gap conditions across both reused Bundesliga match directions. The RNN itself was not retrained. The hybrid uses75% correction and25% RNN conditional live probabilities, followed by the unchanged dead-state fusion and decoder.

| Condition, pooled across two directions | Current014/015 spell F1 | Hybrid017 spell F1 | Current dead F1 | Hybrid dead F1 |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.5054 | 0.5699 | 0.9978 | 0.9978 |
| masked | 0.4156 | 0.4720 | 0.9801 | 0.9802 |
| long_gap | 0.4154 | 0.4652 | 0.9605 | 0.9622 |

Mean across three conditions: **0.4455 → 0.5024**. These figures include a new long-gap condition; do not compare this mean directly with014's two-condition mean.

| Pooled diagnostic | Current | Hybrid017 |
| --- | ---: | ---: |
| missed | 1939 | 1772 |
| false_positive | 716 | 569 |
| merges | 424 | 408 |
| short_under_0_4s | 16 | 13 |

At IoU0.5,017 newly matches290 reference spells but loses123 previously matched spells: net167 fewer misses. These are counts summed across both halves, two matches and repeated clean/stress conditions, not that many unique possessions. Merge counts mean a predicted spell geometrically overlaps multiple same-team reference spells; they are not independently verified merge events.

Post-selection missing-ball audit: missed reference spells touching masked frames fall **817 → 747**. Match success is still measured on the complete sequence; selecting only masked spans would artificially censor and split spells. The audit is descriptive and did not select the model.

## Both match directions

| Validation match | Condition | Current | Hybrid017 | Merge change |
| --- | --- | ---: | ---: | ---: |
| J03WN1 | clean | 0.5028 | 0.5908 | -7 |
| J03WN1 | masked | 0.3914 | 0.4972 | -4 |
| J03WN1 | long_gap | 0.4151 | 0.4799 | -8 |
| J03WMX | clean | 0.5080 | 0.5486 | +1 |
| J03WMX | masked | 0.4395 | 0.4461 | +2 |
| J03WMX | long_gap | 0.4157 | 0.4502 | +0 |

Reverse-direction merges increase slightly despite the pooled reduction;123 existing matches are lost across repeated conditions. This candidate does not establish zero-loss retention, fresh-match performance or challenge accuracy. Neither0.8 spell F1 nor generalisation to arbitrary tracking quality has been reached.

## Preserved rejected experiment016

Before017, we trained a new dead-state head using longer missing-ball augmentation and observation-age features. Features combined original and conservatively reconstructed geometry. Three fixed variants used no team-input reconstruction, gaps up to25frames or up to50frames. Only bracketed gaps within each half with implied ball speed<=50m/s could be filled; no extrapolation. Original observation/repair masks and previous/next observation age remained available to the dead head.

None passed the preregistered guards. Best meanF10.4453 versus current0.4455; total misses1939→1934, false positives716→732, merges424→426. Interpolation alone was not a sufficient improvement. All trained weights and predictions are kept privately under the research run; negative results and code are preserved.

## Protocol and provenance

Existing IDSSE/Sportec data, Bassek et al. (2025), DOI10.1038/s41597-025-04505-y, CC BY4.0. Only Bundesliga first division J03WMX and J03WN1 are used for new training. Each direction trains on one complete match and validates on the other; both halves stay in the same role. Both matches have been used repeatedly, and017 follows inspection of016. This is exploratory development, not nested cross-validation or independent testing. Original source/cache hashes and class counts are in014/015 audits; source licence audit is in docs/data-sources/OPEN_ACCESS_AUDIT_2026-11-12.md. The context RNN remains the separately trained SkillCorner checkpoint.

Target for017 is live home/away from provider BallPossession, restricted to BallStatus=live. Dead labels are excluded from this control fit. No labels, provider status, timestamps or player identities enter features. The new classifier has244 features: original120 and repaired120 geometry/context features, plus original observed flag, filled flag and capped previous/next observation ages. Train augmentations: clean;40% target random20–100frame gaps;40% target random50–300frame gaps. Intervals can overlap into longer contiguous gaps. Both team orders are trained; inference averages both orders exactly. Samples are taken every fifth known live frame.

Histogram gradient boosting:100 iterations,15leaves, minimum40samples/leaf, learning rate0.08, L2=10, seed117, no random-frame validation or early stopping. Input NumPy model stores explicit tree arrays without pickle. Native sklearn and exported probabilities agree on training probes and all validation predictions. Training uses pinned sklearn1.7.2; inference needs only base NumPy/pandas.

Three fixed blends0.25/0.5/0.75 were specified before017 training. Decoder frozen at team penalty12, stop penalty3, dead log-odds weight0.75. Eligibility requires higher three-condition pooled spellF1, no pooled increase in misses/false positives/merges, and no clean fold F1 loss beyond0.01. Only0.75 passes all guards. All trial scores and rejected variants are reported. There is no post-result decoder sweep.

Original tracking has complete ball positions; masking is synthetic and may miss the challenge's biased occlusions, fragmented players and ball false positives. Offline features and repairs use future context within a half. Improvements are provider-specific development evidence. No challenge inputs, video, public ground truth or human-produced challenge annotations were used for this experiment.

## Reproduce and load

Use Python3.11/3.12. Existing external raw XML can be converted with the documented external loader. Recreate the reverse dead estimator with the015 command if needed.

```sh
python -m pip install -r requirements-deadball.txt
python -m controlswitch.control_correction --cache-root data/external/cache --forward-reference models/archive/deadball-014-validation-only.npz --reverse-reference local/deadball-015-reverse-new/dead-estimator.npz --team-model models/rnn-europe-context-001-experimental.npz --out local/control-017-new
```

The016 rejected study uses the same arguments with `python -m controlswitch.gap_possession` and a new output directory. New experiments never overwrite earlier outputs.

```python
from controlswitch.control_correction import load_candidate, infer_candidate
candidate = load_candidate(
    "reports/external/control-017-settings.json",
    "models/rnn-europe-context-001-experimental.npz",
    "models/archive/deadball-014-validation-only.npz",
    "models/archive/control-017-validation-only.npz",
)
labels, probabilities = infer_candidate(candidate, sequence, pitch)
```

The published checkpoint is the forward model trained on J03WMX, not an ensemble of both validation folds. Replay matches all six forward validation outputs exactly, with reference labels removed from model inputs. Hash checks pin all three model assets.

Optional challenge generation with the frozen hybrid:

```sh
python -m controlswitch.submission --data-root data/raw/provided --out local/submission-hybrid017 --settings reports/external/control-017-settings.json --control-model models/archive/control-017-validation-only.npz
```

This command is opt-in. The default submission command still generates014; earlier CSVs, production model and website examples remain preserved. No new challenge CSV was generated or used to select017 in this study. The notebook defaults to saved reports; training and challenge inference remain disabled.

References: [dataset paper](https://doi.org/10.1038/s41597-025-04505-y), [pinned estimator documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
