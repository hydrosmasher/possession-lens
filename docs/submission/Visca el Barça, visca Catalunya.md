# Possession Lens: team control through missing and noisy tracking

**Team:** Visca el Barça, visca Catalunya

**Authors:** Hrishikesh Karmalkar, Basel, Omar, Michail and Paul

**Candidate:** 053 · **Code:** https://github.com/hydrosmasher/possession-lens

## Method

We label each frame as home possession (H), away possession (A), dead ball (D), or outside the published halves (X). Automatic tactical tracking XML supplies team membership; TXT supplies player/ball positions and speeds. Coordinates become centred pitch metres. Frame IDs 1–199319 become CSV frames 0–199318. Halves are processed separately; X is written outside them. No provided video, manual player mapping, homography or Smart Tagging is used by this frozen candidate.

The model separates **who controls live play** from **whether play is live**, then jointly decodes H/A/D. Its 508 identity-free features combine ball/player proximity, relative movement, ball-trajectory changes and collective team shape/motion. These are control/contact proxies, not verified touches. Ball-independent shape features retain evidence during ball gaps. Gaps of at most 50 frames may be interpolated only between visible endpoints with implied speed≤50 m/s; original visibility remains a feature. There is no extrapolation or interpolation across halves, and missing ball observations are not labelled dead automatically.

Conditional H/A evidence combines 50% new GRU,37.5% frozen 040 tree and 12.5% retained context RNN. The GRU uses a learned 64-dimensional projection, LayerNorm/GELU, two bidirectional 64-unit GRU layers, output LayerNorm and locked dropout 0.25. Inference uses 256-frame cores with 64 context frames on each side, matching training. Unused neural D logits are excluded from conditional H/A normalisation. Bidirectional context is offline; this is not a live turnover forecast.

Live/dead evidence combines frozen clean/gap 014 and noise 018 experts. The 029 quality gate uses 27 team-invariant visibility/kinematic features to vary the noise-expert weight between 0.5 and 1.0 in log-odds space; dead odds are scaled 0.75. A balanced boundary verifier adapts the H/A switch-cost base **12** at full strength; stop/restart cost remains 3. No hard possession-duration floor is imposed. Balanced event evidence is not a calibrated touch probability.

## External training only

The GRU continues candidate 052 for 12 fixed iterations of 512 windows. Half its training anchors are sampled uniformly from known-live spells lasting at most three seconds, then uniformly within the spell; these frames receive twice their previous spell-weighted cross-entropy weight. The matched ordinary-continuation control scored 0.624470; focused training scored 0.628641. This gives 52 iterations and 26,624 sampled windows across the 043/052/053 training history, not independent new labels.

The GRU uses AdamW, learning rate 0.0001 for continuation, weight decay 0.001, gradient norm clipping 1 and seed 153. Original 043 training used 24 fixed iterations at learning rate 0.0005 and seed 143;052 added 16 at 0.0001 and seed 152. Xavier input/output and orthogonal recurrent-gate initialisation, spatial reflections, team swaps and fixed tracking corruptions are documented in the numbered protocols. Known-live spell-weighted CE excludes D/unknown/padding/context; auxiliary narrow turnover BCE has weight 0.1 and positive weight capped 10. Scaling fits training observations only. Labels never enter inference.

Submitted forward weights fit **J03WMX** only. Reverse development fits **J03WN1** and validates WMX; forward validates WN 1. Both are first-division Bundesliga IDSSE/Sportec matches, heavily reused for development. Provider BallStatus/BallPossession map mechanically to H/A/D; we created no manual possession annotations. WMX contains 145,967 known 25 Hz frames:85,369 live and 60,598 dead. Licence: CC BY 4.0, [Bassek et al.2025](https://doi.org/10.1038/s41597-025-04505-y).

The retained context RNN uses five existing first-division [SkillCorner](https://github.com/SkillCorner/opendata) games:4039,3749,3518,2068,2440;187,825 labelled native 10 Hz frames, resampled to 25 Hz without filling unknown labels. Its historical MIT release is pinned at revision1ed50b775f93075c2a6d00d0d795f13b7b89709e. These are partial H/A labels, not supervised dead-ball truth. The old context RNN is 24 units, selected iteration 7, dropout 0.15, L2=0.0001 and clipping 5.

Retained 040 uses 200 trees and 136,592 repeated augmented live examples. The 014/018 dead experts use 100 trees each and 58,388/116,776 sampled exposures respectively. The 029 quality gate uses 50 trees and 116,776 exposures labelled by automatic corruption category, not possession. The 031 boundary head uses 14,896 balanced exposures. Repeated windows/conditions do not create independent labels. Exact recipes, source roles and asset hashes are retained in `reports/external/submission-training-lineage.json` and the numbered protocols/configs.

## Evidence and limitations

| Reused external condition | Candidate 053 spell F 1 |
|---|---:|
| clean | 0.7277 |
| masked | 0.6364 |
| long_gap | 0.6109 |
| tracker_noise | 0.5395 |

Overall mean improves **0.615370 → 0.628641** against candidate 052 on the same original four-condition benchmark. Different simulated-error draws after freezing score **0.606253 → 0.622752**. This is not a fresh-match test or a competition score. The headline metric averages official one-to-one spell F 1 across temporal IoU thresholds 0.3–0.7.

Across repeated conditions the final model has 1603 misses,1101 false-positive spells and 369 geometric merges;052 had 1746, 915 and 422. Counts repeat physical possessions. Focused 053 training recovers more short possessions but increases false positives; every category is reported. Refit-boundary 054 alternatives failed their gain/noise requirements and are not silently substituted. Separate older partial-provider diagnostics were weaker and expose transfer risk. The 0.85–0.90 target remains unmet; competition rank is unknown.

No challenge possession truth, video or public score is used for training, tuning or selection. No provided data is uploaded to AI services. Challenge slots are not player identities; named-player and verified-event website examples are separate external demonstrations with a frozen gallery model.

## Local reproduction

Use Python 3.12 and obtain licensed match assets separately:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gru-inference-lock.txt
python -m controlswitch.recurrent_pipeline --data-root /path/to/provided-data --out local/submission-053 --config reports/external/short-spell-053-config.json --filename "Visca el Barça, visca Catalunya.csv"
python evaluation/official/evaluation/validate_submission.py "local/submission-053/Visca el Barça, visca Catalunya.csv"
```

Expected 199,319-row CSV SHA 256: `2f5a445d2c3c10b3126b5bda3a5b9b11050d9a87410f19b7cbc5aaa88b928eeb`. The config pins all model assets and the nested 040 config by SHA 256. CPU inference uses PyTorch 2.9.1, one thread and deterministic algorithms, without a GPU or hosted service. Two isolated source/dependency environments reproduced the CSV identically on the same machine; cross-platform bitwise identity is not claimed. The notebook opens with current evidence and keeps optional training/inference disabled.

The repository remains private pending reviewed public release before submission. Match inputs, feature caches and private outputs are excluded. Earlier models, reports and submission bundles are preserved.

## Separate provider check: a regression to investigate

On the previously reused PSG–Marseille Ligue 1 example, mean partial-label spell F 1 declines **0.300976 → 0.278404** for 052→053. This is a different-provider diagnostic, not a fresh test. Its censored team-possession groups have no verified live/dead labels, so these values cannot be equated to the complete H/A/D benchmark. It was frozen after selection and was not used to refit, tune or reselect the model. The regression exposes transfer risk:053 is a stronger candidate on the current Bundesliga benchmark, not a demonstrated universal improvement.052 remains preserved as a fallback. See reports/external/provider-056-report.json.

The stronger switch-cost checks 055 scored 0.629150(cost 15) and 0.628001(cost 18); neither passed the predeclared improvement requirements. No winner audit was run. The released candidate 053 keeps switch-cost base 12.
