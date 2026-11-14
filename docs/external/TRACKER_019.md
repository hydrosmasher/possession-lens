# Tracker robustness018 and dead-expert blend019

018 trained new team-control and dead-state estimators with synthetic tracking errors on the existing two Bundesliga matches. Every018 replacement failed at least one preregistered guard. The noise-trained dead estimator helped the difficult tracking condition but degraded some previous conditions. Its new control estimator was not retained.

019 followed that result with two fixed log-odds blends,25%/50% of the noise-trained dead expert. It preserves017 team control and all decoder settings. Both blends passed the guards;50% had the higher four-condition mean. A single200-tree numeric NPZ represents the50/50 log-odds average exactly. This adds no inference dependency or external service.

| Pooled external condition | Frozen017 spell F1 | Candidate019 spell F1 | 017 dead F1 | 019 dead F1 |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.5699 | 0.5729 | 0.9978 | 0.9977 |
| masked | 0.4720 | 0.4720 | 0.9802 | 0.9793 |
| long_gap | 0.4652 | 0.4672 | 0.9622 | 0.9628 |
| tracker_noise | 0.2867 | 0.3382 | 0.5000 | 0.7894 |

Four-condition mean spellF1: **0.4484→0.4626**. It includes the new tracker_noise condition and is not comparable to the earlier three-condition or two-condition averages.

| Pooled IoU0.5 diagnostic | 017 | 019 |
| --- | ---: | ---: |
| missed | 2502 | 2458 |
| merges | 551 | 544 |
| false_positive | 1254 | 1100 |
| short_under_0_4s | 16 | 7 |

019 newly matches61 reference spell instances and loses17; the net reduction in misses is44. These totals repeat the same physical possessions across clean/stress conditions. Merges are geometric overlaps, not independent manual event labels.

## Fixed noise and evaluation protocol

Train WMX / validate WN1 and reverse train WN1 / validate WMX; halves stay together. Original short/long-gap seeds are unchanged from017. Tracker noise:40% target missing ball using50–300frame draws,0.25m ball-position jitter and0.5m/s speed jitter;10% target player dropout per slot in25–100frame draws,0.35m player jitter and0.3m/s player-speed jitter; one extra duplicate per team with0.1m jitter. Overlaps may create longer gaps. Missing original observations stay missing. Noise seeds618+i train and818+i validation. Noise has no possession-label inputs and its settings were not measured or tuned on the provided game.

All fitting remains on existing first-division Bundesliga external data under IDSSE/Sportec CC BY4.0.018 uses the same100-iteration,15-leaf gradient boosting recipe, seed118, with clean/short/long/noisy training augmentations. Its H/A classifier uses live labels and team swaps; its dead estimator uses all known H/A/D labels as binary live/dead supervision. No new data download.019 refits nothing: it combines frozen old and noise-trained dead models.

019 eligibility: higher four-condition meanF1; no pooled misses/merges/FP increase; every previous condition in each direction no F1 regression>0.01; tracker-noise F1 in each direction no decline; all per-direction/per-condition deadF1 declines<=0.005. Both fixed weights pass. These tolerance guards do not guarantee every individual score or possession improves. All rejected variants and their exact guards are published.

## Limits and freeze

Both matches have been viewed repeatedly and019 follows018 on the same labels. This is exploratory reused validation, not fresh testing or unbiased cross-validation. Shared provider, clubs and artificial missingness limit transfer. The actual challenge distribution may differ. The public five-minute score was not used to fit, calibrate or select either experiment. No challenge video, labels, tracking or human annotations enter these experiments.

Candidate019 is recommended for the submission freeze after its deterministic replay checks. Legacy production, gallery,017 weights and017 CSV remain preserved. No019 public-window or private accuracy was measured.

## Reproduce

Python3.12 and pinned requirements-deadball.txt for training. Recreate017/015 local reference runs when needed; their weights are also retained privately in local/research. See014–017 guides.

```sh
python -m controlswitch.tracker_robustness --cache-root data/external/cache --reference-root local/control-017-new --forward-dead models/archive/deadball-014-validation-only.npz --reverse-dead local/deadball-015-reverse-new/dead-estimator.npz --team-model models/rnn-europe-context-001-experimental.npz --out local/tracker-018-new
python -m controlswitch.dead_blend --cache-root data/external/cache --reference-root local/control-017-new --forward-dead models/archive/deadball-014-validation-only.npz --reverse-dead local/deadball-015-reverse-new/dead-estimator.npz --noise-root local/tracker-018-new --team-model models/rnn-europe-context-001-experimental.npz --out local/dead-blend-019-new
```

Frozen inference requires only requirements-inference-lock.txt:

```sh
python -m controlswitch.submission --data-root /path/to/provided-data --out local/submission-019 --settings reports/external/deadblend-019-settings.json --dead-model models/archive/deadblend-019-validation-only.npz --control-model models/archive/control-017-validation-only.npz
```

All eight saved forward external validation predictions replay exactly with reference labels absent from inference. Separate source copies and a newly installed base environment produce byte-identical full challenge CSVs. Same machine/architecture only; no cross-platform claim. Proof: reports/external/deadblend-019-reproduction-check.json.

CSV SHA-256: `019e78085b8807b43b6afcc172599509a89fc27eb2ccbd9c5144c9cb1014d961`. Format:199,319 sequential rows; no accuracy measurement. All generated tracking/prediction outputs remain private and ignored.
