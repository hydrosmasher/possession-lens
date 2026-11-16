# More training examples and GRU fine-tuning —083–084

16 November 2026. **Decision: preserve production model053 and the frozen submission CSV.** No tested candidate passes every acceptance check. Four newly trained diagnostic GRUs are archived separately.

## Data search and what was actually added

No new anonymous, licensed top-five first-division match with continuous player/ball tracking and usable possession reference was verified for this run. The source audit is `reports/external/data-access-083-catalog.json`; it is a search record, not proof that no other suitable source exists.

- IDSSE/Sportec: the original publication has two first-division and five second-division matches. The PySport mirror is not an independent set of new first-division matches. We retain WMX/WN1 and exclude second division under the user's league restriction. Source: https://www.nature.com/articles/s41597-025-04505-y
- Historical SkillCorner: replay the five previously audited MIT training games4039/3749/3518/2068/2440. Their partial H/A labels do not provide full H/A/D evaluation. Four other previously seen historical matches are not added to this fitting pool. Source revision: https://github.com/SkillCorner/opendata/tree/1ed50b775f93075c2a6d00d0d795f13b7b89709e
- IMPECT: public Bundesliga2023/24 event/KPI/lineup data is not continuous all-player/ball tracking with frame possession labels. It was not downloaded or used. Source: https://github.com/ImpectAPI/open-data
- SoccerTrack v2: the official project describes university-level amateur matches, outside the requested league scope. Not used. Source: https://atomscott.github.io/SoccerTrack-v2/
- SoccerNet GSR2024: public athlete tracking clips exist, but the primary benchmark excludes the ball from pitch reconstruction and ignores ball detections. Suitable continuous ball tracking/possession labels were not verified. NDA-gated ball-action tasks were not accessed; a newer catalogue lead was not verified usable. Sources: https://huggingface.co/datasets/SoccerNet/SN-GSR-2024 and https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.pdf

**Actual expansion:** eight new synthetic tracking-error views of the existing two Bundesliga matches: two additional random noise draws for each half of each match, seeds18618/28618 plus half index. These add observation diversity, not matches or independent labels. The earlier four training conditions are also retained. Raw arrays stay in ignored local storage.

Across the two opposite-match experiments there are seven existing matches in total. Each fitted model sees only its corresponding Bundesliga training match plus the five fixed historical weak matches; it never fits the opposite Bundesliga evaluation match. All full-H/A/D evaluation remains on two heavily reused games and error draws, not fresh independent testing.

##083: actual GRU continuation

Both branches start from the corresponding frozen053 checkpoint and scaler: two bidirectional GRU layers,64 hidden units,64 projection,508 features, dropout0.25. No additional layers or scaler fitting. Training uses80% strong windows and20% historical replay; strong entries now include extra independent tracking-error views. Short known segments receive the existing doubled loss weight and focused sampling. Historical unknown labels remain excluded, with no fabricated dead-ball labels or event targets.

Training is fixed at12iterations×512windows per model, batch32, AdamW learning rate0.0001, weight decay0.001, gradient clipping1, seed183+fold. Both branches have the same samples, dropout draws, ordinary control loss and auxiliary boundary loss. No held-out score selects epochs or checkpoints.

- `replay_ce`: matched ordinary continuation control.
- `replay_continuity`: adds0.1×adjacent symmetric KL only where consecutive training labels belong to the same known live team. Genuine label changes, dead ball, unknown frames and padding receive no consistency penalty. There is no minimum possession duration or reference-label input at inference.

There are four fitted models: two branches in each of two opposite-match folds, each consuming6144training windows. Learning traces record component losses, gradient norms, timing and sampled sources. These are training diagnostics, not proof of generalisation or an independent overfitting estimate.

Inference updates the GRU half of the existing50/50 probability fusion using the exact arithmetic residual. Model053's D labels and boundary costs stay fixed. Missing-ball H/A labels are constrained to053. Existing081 novel receipt spans are constrained to their predicted team, but their flanks can change so fragments may reconnect. These anchors come from predictions, never reference labels. A same-anchor no-training control isolates the effect of fine-tuning.

##084: protect already-detected short spells

083 retained every one of the46 newly recovered matches but lost six other previously matched short possessions. A separately preregistered decoder-only follow-up protects every081 predicted H/A spell no longer than150frames and its immediate flanks. The bound follows from IoU>=0.5: a predicted interval matching a <=75frame reference cannot exceed150frames in a fully scored run. No reference label chooses which intervals are protected. Missing/D constraints remain.

This argument has a limitation at unscored reference-run boundaries, so exact external short-spell matching is still checked. The guard also preserves false predicted short spells and limits opportunities to remove false positives. Both frozen083 models and a matched no-training control are evaluated under the new constraint. No retraining or checkpoint search occurs in084.

## Results

| Method | Mean spell F1 | Missed | False positives | Matched <=3s spells | Lost previous short matches |
|---|---:|---:|---:|---:|---:|
| Frozen053 | 0.628641 | 1603 | 1101 | 787 | 0 |
| Previous research081 | 0.632230 | 1561 | 1133 | 824 | 0 |
| continuity-refinement-083: anchors_control | 0.632230 | 1561 | 1133 | 824 | 0 |
| continuity-refinement-083: replay_ce | 0.632936 | 1554 | 1126 | 828 | 6 |
| continuity-refinement-083: replay_continuity | 0.632989 | 1554 | 1126 | 828 | 6 |
| short-preservation-084: short_anchors_control | 0.632230 | 1561 | 1133 | 824 | 0 |
| short-preservation-084: replay_ce | 0.631793 | 1558 | 1140 | 826 | 0 |
| short-preservation-084: replay_continuity | 0.631793 | 1558 | 1140 | 826 | 0 |

| Method | WN1 noisy spell F1 | WMX noisy spell F1 |
|---|---:|---:|
| Frozen053 | 0.552804 | 0.526092 |
| continuity-refinement-083: anchors_control | 0.556705 | 0.524973 |
| continuity-refinement-083: replay_ce | 0.560677 | 0.520298 |
| continuity-refinement-083: replay_continuity | 0.561099 | 0.520298 |
| short-preservation-084: short_anchors_control | 0.556705 | 0.524973 |
| short-preservation-084: replay_ce | 0.556705 | 0.523860 |
| short-preservation-084: replay_continuity | 0.556705 | 0.523860 |

The best083 training branch reaches0.632989 versus081at0.632230, reducing both misses and false-positive spells by7. It retains all46 newly recovered matches. However, aggregate short matches rising824to828 hides churn: six other short matches are lost while others are gained. Eight previously matched053spells suffer additional loss versus070. WMX noisy tracking regresses from053's0.526092to0.520298. False positives1126 remain above053's1101. Therefore it is not an acceptable replacement.

The continuity penalty changes the mean by only about0.000053 versus the matched CE branch at this fixed strength; this is not evidence of a statistically established improvement. More data views and replay modestly improve the aggregate score but do not solve robustness.

084 preserves all previous short matches and all46recoveries, but scores0.631793 with1140false positives, worse than the no-training control. It is rejected too. No different-error-draw promotion audit is run for candidates already failing development checks. Neither a favourable fold nor a hand-picked subset of model edits is deployed.

## Reproduce and review

The compact judge notebook displays both new studies without running inference/training by default. The research notebook contains full alternatives, learning summaries and guarded optional run commands. Default production053 inference and CSV remain unchanged.

Full research reproduction requires the documented Python3.12 research environment, the licensed input caches and existing ignored transfer-corpus052/053/055/069/070/077/081 data. Run in a separate research checkout; output folders refuse overwrite:

```sh
python -m controlswitch.continuity_refinement_experiment --root . --cache-root /path/to/licensed/cache --corpus-root /path/to/local/transfer-corpus-052
python -m controlswitch.short_preservation_experiment --root . --cache-root /path/to/licensed/cache
python -m unittest discover -s tests -p test_continuity_refinement.py -v
python -m unittest discover -s tests -p test_short_preservation.py -v
```

Numeric checkpoints and their hashes are in `models/archive/gru-continuity083-manifest.json`. They are research assets, not production defaults. Raw match inputs, additional generated feature arrays and per-frame predictions remain ignored underlocal/. No supplied challenge match, video, ground truth or public score was used to train, tune or select these candidates.
