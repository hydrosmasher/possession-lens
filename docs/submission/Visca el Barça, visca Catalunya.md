# Possession Lens: robust H/A/D possession segmentation

**Registered team:** Visca el Barça, visca Catalunya

**Authors:** Hrishikesh Karmalkar, Basel, Omar, Michail and Paul. **Candidate:**032. **Code:** https://github.com/hydrosmasher/possession-lens (public-release review pending).

## Approach

We estimate home control(H), away control(A), and dead ball(D) for every tactical-video frame. The method separates who controls live play from whether play is live, then jointly decodes the three states. It runs locally without a GPU, hosted model or network request.

Team evidence combines the frozen24-unit context RNN (120 geometry/context features) with an external-trained100-tree H/A classifier. Its296 features retain the earlier244-feature correction and add ball/player relative movement, soft geometric interaction evidence and multiscale past/future team-association memory. We average both team permutations, then blend75% classifier and25% RNN live probabilities. Interaction features do not establish verified touches.

Live/dead evidence combines frozen clean/ball-gap014 and tracking-noise018 experts. A50-tree observation-quality gate learns27 team-invariant kinematic/visibility/duplicate-track features from automatic external corruption categories, not possession labels. It sets the noise-expert weight between0.5 and1.0 in log-odds space. Dead odds are scaled0.75 before fusion. A balanced116-feature boundary head refitted on the frozen030 emissions adapts H/A switch cost12 at strength1.0; stop/restart cost3 remains fixed. Offline decoding resets each half. All settings were selected externally; no new basic RNN, LSTM or GRU is claimed.

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

WMX contains145,967 known25Hz frames:85,369 live and60,598 dead. The032 H/A classifier trains on136,592 augmented live examples with four observation conditions and team swapping; bounded inverse-square-root spell-duration weights give short possessions greater influence (train-half duration median, raw clip[0.5,3], then unit live mean); old014 dead model on58,388 augmented examples; noise018 dead model on116,776 examples. These are sampled/repeated training exposures, not unique new labels. Boosting settings:100 iterations,15leaves, learning rate0.08, minimum40samples/leaf,L2=10,no early stopping.018 noise is fixed synthetic ball gaps/jitter, player gaps/jitter and two duplicate tracks; training and validation seeds differ.029 trains the quality gate on116,776 automatically augmented exposures (50iterations,7leaves,minimum80samples/leaf).028 retains the same100-iteration training recipe;031 refits the boundary head on14,896 balanced sampled WMX exposures (7,448positive/7,448negative),25logistic epochs,seed131,L2=0.001, using train-only standardisation. All remaining030 components stay frozen; nothing fits on the challenge.

Complete label counts, source hashes, model hashes and split history are in reports/external/submission-training-lineage.json and014–019 reports. Earlier second-division models remain historical comparators; they are not components of030. Sources: [SkillCorner](https://github.com/SkillCorner/opendata), [IDSSE/Sportec publication](https://doi.org/10.1038/s41597-025-04505-y).

## External evidence and limitations

| Reused external condition |031 spell F1 |032 spell F1 |
|---|---:|---:|
| clean | 0.6248 | 0.6434 |
| masked | 0.5284 | 0.5325 |
| long_gap | 0.4968 | 0.5104 |
| tracker_noise | 0.4388 | 0.4585 |

Mean spellF1 improves0.522219→0.536213. Across repeated conditions, misses2175→2110, false positives1031→1019 and geometric merges482→472.163 reference-spell instances are newly recovered and98 previously matched instances are lost. Counts repeat physical possessions across conditions. Ordinary frame weighting with the same added noise examples scores0.507568 and fails guards. All variants, settings and condition/dead-state regression checks were fixed before outcomes.

Weights were frozen before a different-observation-draw audit:0.513297→0.532568 on the same games. This is not independent-match testing. Both games have been extensively reused, synthetic errors differ from real occlusions, and provider labels/tracking are correlated. Offline inference uses future observations. No challenge video, possession truth or public score was used for training, tuning or selection.0.7 spellF1 and top5% competition rank are goals, not established results. Rejected033 and035 studies do not alter the frozen032 submission. Earlier releases remain reproducible.

## Exact reproduction

Use Python3.12. Obtain licensed match assets separately, keeping them out of the repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference-lock.txt
python -m controlswitch.ranked_pipeline --data-root /path/to/provided-data --out local/submission-032 --config reports/external/spell-training-032-config.json --filename "Visca el Barça, visca Catalunya.csv"
python evaluation/official/evaluation/validate_submission.py "local/submission-032/Visca el Barça, visca Catalunya.csv"
```

Expected CSV SHA256: `f11559b18434dee7c82de966890d4b4b66b7d0b5cec0101361e6a6779b9e9e85`. Two separate source copies and dependency environments reproduced it on the same machine; all16 saved boundary-head/decoder predictions across both directions replay exactly. Six model hashes are enforced by the config. Training libraries are unnecessary at inference; no pickle/network is used. CSV has199,319 ordered frame,label rows. The prepared CSV and document use the registered team name exactly. The code repository must be public at submission, with licensed assets excluded; public-release review remains pending. Previous work is retained for reproducibility.
