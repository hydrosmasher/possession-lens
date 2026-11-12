# Expanded-data temporal-context experiment

The previous spell-weighted loss and boundary heads did not improve validation. This experiment changes the information presented to the recurrent model and expands its training pool, rather than adding layers or repeating those rejected changes.

## Fixed split

| Role | Matches |
|---|---|
| Training | 4039 Manchester City–Liverpool; 3749 Dortmund–Bayern; 3518 Juventus–Inter; **2068 Inter–Juventus, 6 October 2019**; **2440 Liverpool–Manchester City, 10 November 2019** |
| Validation | 2269 PSG–Marseille |
| Fresh test | **2841 Barcelona–Real Madrid, 18 December 2019, La Liga** |
| Excluded viewed tests | 3442 Real Madrid–Barcelona; 2417 Bayern–Dortmund |

All matches are top-five European domestic leagues. The fresh test's clubs are absent from this training pool, but they appeared in a previously scored test, so this is not a completely blind unseen-team research programme. Neither previous test is used for training, checkpoint selection, calibration or decoder tuning.

Three additions are imported from the already downloaded official historical SkillCorner ZIP, pinned at revision `1ed50b775f93075c2a6d00d0d795f13b7b89709e`. File sizes and Git blob hashes are checked against the official GitHub directory. Import copies opaque bytes without parsing test labels; the split was written before extraction. MIT attribution is retained. The new training matches have approximately 56.6–64.2% provider-labelled playing frames per half.

## Different approach: context plus a data-only ablation

Two models are trained from scratch on exactly the same expanded five-match pool:

- **Expanded instantaneous RNN:** the original 30 geometry and motion features. This checks what extra data alone contributes.
- **Expanded temporal-context RNN:** 120 features comprising the same instantaneous inputs, centred means over ±0.48 seconds and ±2 seconds, and a contrast of ±0.48-second means sampled at ±1 second. This gives the network observation evidence before and after a possible short possession, which may help distinguish a brief opposing spell from tracking noise.

Features use tracking only. Labels never enter feature functions. Future tracking is used, so this is **offline match analysis**, not a live forecast. Context can span unknown-label intervals without filling or grading their missing labels. It is not guaranteed to recover a possession that has no informative tracking evidence.

Both models have one recurrent layer with 24 units, standard known-frame cross-entropy, team-swap and fixed contiguous ball-dropout augmentation, dropout 0.15, L2 0.0001, Adam, gradient clipping at norm 5, 128-frame training windows and 16-window batches. There are at most 16 sampled iterations, each using at most 1,024 windows. An iteration is not a full epoch. Training-window state resets; inference carries recurrent state within a half. Neither rejected spell weighting nor a boundary head is reused.

## Selection and one-time testing

All checkpoints are compared by mean clean/dropout **validation possession-spell F1**, with decoder penalties 3, 12, 24 and 48. A candidate passes the switching safeguard if pooled false-positive spells at IoU 0.5 are at most 110% of the previous European model and pooled sub-0.4-second pulses are at most five higher. An eligible checkpoint is preferred. If none passes, the best new checkpoint is frozen as an explicitly rejected experimental diagnostic; the old model is never silently substituted and labelled a new candidate. Ties retain the earlier checkpoint and smaller penalty.

Both validation-selected family checkpoints, the primary candidate, the baseline window and comparator hashes are saved before test loading. Their fresh-test scores and missed/merged-spell diagnostics are all reported, but test ablations cannot be used to change the primary model selected on validation. Test dropout seed is 500. No post-test tuning is permitted. Production defaults are not automatically changed by this partial-label experiment.

## Interpretation and remaining limits

Provider-generated H/A labels are partial weak supervision, not independently annotated full-match truth. Null labels stay X, have no training loss and split graded runs; dead-ball labels are never invented. Unknown gaps censor spells and can inflate F1. Tracking interpolation to the common 25 Hz interface does not create independent evidence. Compare models on the same test; a score from a different match does not establish an algorithm gain.

The report includes clean/dropout spell F1, missed reference spells and false positives at IoU 0.5, duration-bin recall, short pulses and possible merges. A possible merge is a predicted spell overlapping multiple same-team reference spells, a descriptive geometric count rather than a confirmed event. F1 gains and miss/merge tradeoffs must both be disclosed. Repeatedly reused PSG–Marseille validation may be overfit. One fresh match is limited evidence, and the 0.8–0.9 target is not guaranteed.

After this experiment, all nine matches in this particular historical release have been used for training, validation or testing. Another genuinely fresh test requires a different licensed release; reversing teams, resampling frames or replaying a viewed match does not make it fresh.

## Reproduce locally

Open `notebooks/01_Possession_Lens.ipynb`. Default cells display aggregate results and run local checks. Training and import remain disabled. The optional importer verifies only the three additions and never parses test labels. Reproduction explicitly marks the test as already viewed. Raw inputs and complete run history remain under ignored `data/` and `local/research/` folders; source archives and GitHub contain no match data or private challenge outputs.

## Recorded outcome

Validation selected the context RNN at iteration 7, penalty 48. Mean fresh-test spell F1 was **0.6286**, versus **0.5804** for the previous European RNN and **0.5846** for the expanded-data instantaneous ablation. Clean F1 was 0.6507 and dropout F1 0.6065. Clean misses fell 125→115, false positives 49→37 and possible merges 29→27; dropout misses/false positives fell by 8 each while possible merges were unchanged. Short pulses decreased in both conditions. This is a same-match gain, not a comparison of scores across different tests.

The new checkpoint is recommended for experimental European transfer. Production defaults remain unchanged because dead-ball recognition and complete-match spells are not validated by these labels. Misses and merges remain substantial; the 0.8–0.9 target has not been reached. No test-informed tuning or ablation-based reselection was performed.
