# Possession Lens: robust H/A/D possession segmentation

**Registered team:** Visca el Barça, visca Catalunya

**Authors:** Hrishikesh Karmalkar, Basel, Omar, Michail and Paul. **Candidate:**019. **Code:** https://github.com/hydrosmasher/possession-lens (public-release review pending).

## Approach

We estimate home control(H), away control(A), and dead ball(D) for every tactical-video frame. The method separates who controls live play from whether play is live, then jointly decodes the three states. It runs locally without a GPU, hosted model or network request.

Team evidence combines a frozen24-unit tanh RNN with an external-trained gradient-boosted H/A classifier. The RNN uses120 multiscale geometry/context features: visibility, ball/player kinematics, proximity and collective position/movement. The correction uses244 features, adding short bracketed interpolation and observation-age information. It blends75% classifier and25% RNN conditional live probabilities. Averaging both team orders enforces exact home/away permutation symmetry.

Live/dead evidence combines two externally trained100-tree classifiers: an earlier clean/ball-gap estimator and a tracking-noise-trained estimator. Candidate019 averages their log odds50/50, exported as one200-tree numeric NPZ. A fixed0.75 log-odds scaling is applied before combining dead probability with conditional team probabilities. Offline Viterbi uses H/A switch cost12 and stop/restart cost3, separately within each half. These settings were selected on external data.

## Assets and possession rules

Inference uses the tactical tracking XML for team membership and TXT for player/ball coordinates and speeds. ATD IDs1…199319 become CSV frames0…199318; normalised positions become centred pitch metres. Published half bounds reset inference. Outside halves we writeX. Inside halves the decoder returns H/A/D, with no fabricated contested state.

A missing ball is missing evidence, not a dead-ball label. Short ball gaps up to50frames may be interpolated only between observed endpoints with implied speed<=50m/s; never extrapolated, and never across a half. Original visibility remains an explicit feature. The learned sequence model can retain team control through loose-ball periods rather than switching on each nearest-player touch. D separates possessions even when the same team restarts. These are statistical proxies for controlled possession, not verified contacts or referee decisions.

Tactical Smart Tagging and homography are not used in this frozen candidate. They were considered as supporting evidence, but no matched external validation justified adding tag-based corrections. Broadcast tracking/video require alignment and were left out. No provided-game video is read; no manual tracking, name or event corrections are used. Challenge slots are not player identities; named-player and verified-event website examples are external demonstrations only.

## Training data and labelling

All training/tuning uses external games. Our team created no manual possession annotations. SkillCorner supplies partial possession groups, treated as weak supervision; null remains unknown. Its historical MIT release is pinned at revision1ed50b775f93075c2a6d00d0d795f13b7b89709e. The frozen RNN training pool is:

| ID | Match | Labelled native10Hz frames |
| --- | --- | ---: |
| 4039 | Manchester City–Liverpool Football Club | 36,429 |
| 3749 | Borussia Dortmund–FC Bayern Munchen | 41,422 |
| 3518 | Juventus–Inter Milan | 38,324 |
| 2068 | Inter Milan–Juventus | 35,284 |
| 2440 | Liverpool Football Club–Manchester City | 36,366 |

Total:187,825 labelled native frames across five first-division matches; native observations are resampled to25Hz without inventing labels in unknown gaps. PSG–Marseille2269 was reused validation. Augmentation swaps teams and masks ball observations. The retained RNN checkpoint is context experiment001, selected iteration7. It was trained from scratch using sampled128-frame windows, dropout0.15, L2=0.0001, Adam and gradient clipping5. Historical partial H/A scores do not validate dead ball.

New H/A and dead-state components use the two existing first-division Bundesliga IDSSE/Sportec games J03WMX and J03WN1. Sportec/DFL supplies BallStatus and BallPossession; our parser mechanically maps dead status toD and live ownership toH/A. Data licence:CC BY4.0, Bassek et al.(2025), DOI10.1038/s41597-025-04505-y. The submitted forward weights train on WMX, both halves; WN1 is reused validation. Reverse-direction experiments train WN1 and validate WMX for development robustness. Both had been viewed previously; no independent-test claim is made.

WMX contains145,967 known25Hz frames:85,369 live and60,598 dead. The017 H/A classifier trains on102,444 augmented live examples; old014 dead model on58,388 augmented examples; noise018 dead model on116,776 examples. These are sampled/repeated training exposures, not unique new labels. Boosting settings:100 iterations,15leaves, learning rate0.08, minimum40samples/leaf,L2=10,no early stopping.018 noise is fixed synthetic ball gaps/jitter, player gaps/jitter and two duplicate tracks; training and validation seeds differ.019 blends frozen experts rather than fitting another model.

Complete label counts, source hashes, model hashes and split history are in reports/external/submission-training-lineage.json and014–019 reports. Earlier second-division models remain historical comparators; they are not components of019. Sources: [SkillCorner](https://github.com/SkillCorner/opendata), [IDSSE/Sportec publication](https://doi.org/10.1038/s41597-025-04505-y).

## External evidence and limitations

| Pooled development condition | Frozen017 spell F1 | Candidate019 spell F1 |
| --- | ---: | ---: |
| clean | 0.5699 | 0.5729 |
| masked | 0.4720 | 0.4720 |
| long_gap | 0.4652 | 0.4672 |
| tracker_noise | 0.2867 | 0.3382 |

Across repeated clean/stress evaluations, misses2502→2458, geometric merges551→544 and false-positive spells1254→1100.019 recovers61 reference-spell instances and loses17 previously matched instances. Counts repeat physical possessions across conditions. Guards protect prior conditions within stated tolerances; they do not establish zero regression. Model choice used external spellF1 and diagnostics only. The provided public reference is reserved for checking a frozen pipeline; no019 score on it or the private match is claimed.

Both Bundesliga matches have been reused extensively;019 follows inspection of018. Provider tracking and labels are not independent, synthetic errors differ from real occlusions, and the full challenge may perform differently. Offline context uses future observations, so this is not a live forecasting model. No0.8 spellF1 or full-match accuracy claim is supported. With more time we would obtain independently labelled matches, test different providers, and improve stoppage/restart evidence under naturally biased missingness.

## Exact reproduction

Use Python3.12. Obtain the licensed match assets separately; do not commit them. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference-lock.txt
python -m controlswitch.submission --data-root /path/to/provided-data --out local/submission-019 --settings reports/external/deadblend-019-settings.json --dead-model models/archive/deadblend-019-validation-only.npz --control-model models/archive/control-017-validation-only.npz --filename "Visca el Barça, visca Catalunya.csv"
python evaluation/official/evaluation/validate_submission.py "local/submission-019/Visca el Barça, visca Catalunya.csv"
```

Expected CSV SHA-256: `019e78085b8807b43b6afcc172599509a89fc27eb2ccbd9c5144c9cb1014d961`. Two separate source copies and independently installed dependency environments reproduced this exact file on the same machine. Input/model/settings hashes are recorded in the private run manifest. CSVs contain199,319 ordered frame,label rows and stay in ignored local folders. Training libraries are unnecessary for inference; models load without pickle. The prepared CSV and this document use the registered team name exactly, including accents and punctuation. The repository must be public at submission, with licensed match assets excluded.
