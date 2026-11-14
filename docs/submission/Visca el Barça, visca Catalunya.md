# Possession Lens: robust H/A/D possession segmentation

**Registered team:** Visca el Barça, visca Catalunya

**Authors:** Hrishikesh Karmalkar, Basel, Omar, Michail and Paul. **Candidate:**040. **Code:** https://github.com/hydrosmasher/possession-lens (public-release review pending).

## Approach

We estimate home control(H), away control(A), and dead ball(D) for every tactical-video frame. The method separates who controls live play from whether play is live, then jointly decodes the three states. It runs locally without a GPU, hosted model or network request.

Team evidence combines the frozen24-unit context RNN (120 geometry/context features) with an external-trained200-tree H/A classifier. Its508 features combine control/memory geometry, relative ball/player velocity, interception/acceleration proxies and ball-independent team shape, movement and field occupancy. Team attack direction is inferred automatically from half-wide extreme-player team votes; ambiguous votes become0. Stationary heading changes are masked. These are geometric proxies, not confirmed touches. We average both team permutations, then blend75% classifier and25% RNN live probabilities. Interaction features do not establish verified touches.

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

WMX contains145,967 known25Hz frames:85,369 live and60,598 dead. The040 H/A classifier trains on136,592 augmented live examples with four observation conditions and team swapping; bounded inverse-square-root spell-duration weights give short possessions greater influence (train-half duration median, raw clip[0.5,3], then unit live mean); old014 dead model on58,388 augmented examples; noise018 dead model on116,776 examples. These are sampled/repeated training exposures, not unique new labels. Control boosting settings:200 iterations,15leaves, learning rate0.08, minimum40samples/leaf,L2=10,no early stopping.018 noise is fixed synthetic ball gaps/jitter, player gaps/jitter and two duplicate tracks; training and validation seeds differ.029 trains the quality gate on116,776 automatically augmented exposures (50iterations,7leaves,minimum80samples/leaf).040 retains032 sample weights, augmentation and seed132, expanding the feature set and training200 iterations;031 refits the boundary head on14,896 balanced sampled WMX exposures (7,448positive/7,448negative),25logistic epochs,seed131,L2=0.001, using train-only standardisation. All remaining030 components stay frozen; nothing fits on the challenge.

Complete label counts, source hashes, model hashes and split history are in reports/external/submission-training-lineage.json and014–019 reports. Earlier second-division models remain historical comparators; they are not components of030. Sources: [SkillCorner](https://github.com/SkillCorner/opendata), [IDSSE/Sportec publication](https://doi.org/10.1038/s41597-025-04505-y).

## External evidence and limitations

| Reused external condition |037 spell F1 |040 spell F1 |
|---|---:|---:|
| clean | 0.6748 | 0.6802 |
| masked | 0.5542 | 0.5858 |
| long_gap | 0.5296 | 0.5680 |
| tracker_noise | 0.4767 | 0.5084 |

Mean spellF1 improves0.558833→0.585574. Repeated-condition misses1996→1870 and geometric merges451→431, while false positives rise997→1007.278 spell instances are newly recovered and152 previously matched instances are lost. Short predicted pulses rise18→21. Counts repeat physical possessions across conditions. The100-tree joint ablation scores0.583619. Training/variants and condition/dead-state regression checks were fixed before outcomes.

After freezing, different observation draws improve0.553212→0.588338 versus037 on the same games. Both games are heavily reused; this is not independent-match testing.040 improves every original condition/fold versus038 as a further development comparison. Synthetic errors differ from real occlusions and provider tracking/labels are correlated. Separate reused PSG–Marseille partial-provider diagnostics score about0.31 with false-dead predictions on known-team frames; censored weak labels differ from full Sportec H/A/D truth, but expose transfer risk.

Offline context uses future observations. No challenge video, possession truth or public score was used for training, tuning or selection.0.7 four-condition spellF1 and top5% rank remain goals. Rejected039/041 and pending042–044 do not alter040. Earlier work remains reproducible.

## Exact reproduction

Use Python3.12. Obtain licensed match assets separately, keeping them out of the repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference-lock.txt
python -m controlswitch.ranked_pipeline --data-root /path/to/provided-data --out local/submission-040 --config reports/external/joint-040-config.json --filename "Visca el Barça, visca Catalunya.csv"
python evaluation/official/evaluation/validate_submission.py "local/submission-040/Visca el Barça, visca Catalunya.csv"
```

Expected CSV SHA256: `8109c7508182c907ede73ac19868d220b36b41c9cb5b369258308345af40298b`. Two separate source copies and dependency environments reproduced it on the same machine; all16 saved boundary-head/decoder predictions across both directions replay exactly. Six model hashes are enforced by the config. Training libraries are unnecessary at inference; no pickle/network is used. CSV has199,319 ordered frame,label rows. The prepared CSV and document use the registered team name exactly. The code repository must be public at submission, with licensed assets excluded; public-release review remains pending. Previous work is retained for reproducibility.
