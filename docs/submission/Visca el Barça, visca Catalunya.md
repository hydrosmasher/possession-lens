# Possession Lens: robust H/A/D possession segmentation

**Registered team:** Visca el Barça, visca Catalunya

**Authors:** Hrishikesh Karmalkar, Basel, Omar, Michail and Paul. **Candidate:**052. **Code:** https://github.com/hydrosmasher/possession-lens (public-release review pending).

## Approach

We estimate home control(H), away control(A), and dead ball(D) for every tactical-video frame. The method separates who controls live play from whether play is live, then jointly decodes the three states. It runs locally without a GPU, hosted model or network request.

Team control combines the retained040 model with a new recurrent model.040 mixes a24-unit context RNN (trained on five external SkillCorner games) with a200-tree classifier using508 identity-free relative/collective motion and geometry features. Attack direction is calculated automatically per half from extreme-player team votes; ambiguous votes become0. Stationary heading changes are masked. These are contact proxies, not verified touches.

The new GRU uses the same508 features, a64-dimensional learned projection, two bidirectional64-unit recurrent layers, projection/output LayerNorm and locked dropout0.25. It trains from scratch for24 fixed iterations ×512 sampled windows, AdamWlr0.0005/weightdecay0.001, gradient norm clipping1, Xavier input/output and orthogonal recurrent gates.256-frame loss cores have64 context frames each side, matching overlap-save inference. Spell-weighted H/A CE excludes dead/unknown/context/padding labels; an auxiliary narrow turnover BCE (+-2frames) has weight0.1. All normalisation fits training observations only; labels are never model inputs.

The selected live-control mixture is50%newGRU+50%040 (overall50%GRU,37.5%tree,12.5%oldRNN). We average team permutations and normalise only H/A neural probabilities; the GRU's unusedD logit is excluded. Dead-ball experts remain separate.

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

WMX contains145,967 known25Hz frames:85,369live and60,598dead. The040 control classifier uses136,592 repeated augmented live examples,200trees/15leaves/learning rate0.08/minimum40samples per leaf/L2=10/seed132/no early stopping. Raw inverse-square-root training-spell weights are clipped[0.5,3], then normalised to unit live mean; final normalised weights can exceed3. Four observation conditions and team swapping are used.

The retained014 clean/gap dead expert uses58,388 sampled exposures;018 noise dead expert116,776. Each has100trees trained onWMX.029's50-tree quality gate uses116,776 exposures labelled by automatically generated corruption category, not possession.031's frozen linear boundary head uses14,896 balanced sampled exposures fromWMX. Exact recipes, source/model hashes and historical split decisions are retained in the numbered reports and submission-training-lineage.json.

Synthetic conditions add short/long ball gaps, ball/player jitter, player dropouts and two duplicate tracks, with separate training/evaluation seeds. They do not establish performance under all real tracking failures. Only first-division external data supplies final components; older lower-division comparators are preserved separately. Sources: [SkillCorner historical release](https://github.com/SkillCorner/opendata), [IDSSE/Sportec publication](https://doi.org/10.1038/s41597-025-04505-y).

## External evidence and limitations

Candidate052 is the final checkpoint for14 November2026. It continues043 for16 fixed iterations ×512 windows (8192 additional sampled windows;40 iterations/20480windows including043), using known-live anchored sampling, coherent x/y field reflections and team swaps. Learning rate0.0001, AdamWweightdecay0.001, clipping1,seed152; architecture,scaler,dead/quality experts and50%GRU mixture stay unchanged. Forward selected weights useWMX only; reverse usesWN1. Two branches adding the five existing SkillCorner games were compared, but the winning spatial_more branch used no weak-game windows. Extra data did not automatically produce the best result.

Four-condition development mean0430.609188→0520.615370. Clean0.712949, short gaps0.621924, long gaps0.592691, tracking noise0.533918. Repeated-condition misses1777→1746, false positives925→915, merges425→422; counts repeat physical possessions across conditions. Frozen different-draw audit0430.602665→0520.609770 passed. The same two matches are heavily reused; this is not independent-match or competition performance. Mean0.7 and top5%rank remain unachieved/unverified.

Earlier043 improved0400.585574→0.609188. Separate older031/032 partial-provider checks score about0.31 and expose transfer risk;052 has no independent-provider claim. Offline bidirectional context uses future observations. All licensed source and retained040 training counts above remain applicable.

## Exact reproduction

Use Python3.12. Obtain licensed match assets separately, keeping them out of the repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gru-inference-lock.txt
python -m controlswitch.recurrent_pipeline --data-root /path/to/provided-data --out local/submission-052 --config reports/external/transfer-052-config.json --filename "Visca el Barça, visca Catalunya.csv"
python evaluation/official/evaluation/validate_submission.py "local/submission-052/Visca el Barça, visca Catalunya.csv"
```

Expected CSV SHA256: `9c1eef2240eef12b72d2f79ce4f5fbb4bd29678cba2975a5d80cf87a683dd459`. Two separate source copies and dependency environments reproduced it on the same machine; all16 saved boundary-head/decoder predictions across both directions replay exactly. Seven model hashes plus the nested040config hash are enforced. CPU inference uses pinned PyTorch2.9.1 with1thread/deterministic algorithms; no GPU, pickle or network request is used. CSV has199,319 ordered frame,label rows. The prepared CSV and document use the registered team name exactly. The code repository must be public at submission, with licensed assets excluded; public-release review remains pending. Previous work is retained for reproducibility.
