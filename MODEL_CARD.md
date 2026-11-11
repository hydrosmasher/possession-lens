# Possession Lens: external pilot and development models

## Current team direction: recurrent starter (10 November 2026)

The new starter is a genuine NumPy tanh RNN with 24 hidden units and a three-state softmax output. It uses current observation features at 25 Hz and truncated backpropagation through time with 128-frame chunks. Hidden state carries between chunks and resets between periods. Standardisation fits external training observations only.

The first five-epoch run used the previously parsed external Sportec cache, with the original chronological development split. Validation spellF1 was 0.3408. The previously viewed test half was not scored again. This is a starter, not an optimised model or evidence of an improvement over the earlier classical experiments.

The frozen checkpoint generated a full-length CSV on the provided automatic tracking using inference only. No competition output was scored against ground truth or checked against video. Team percentages are model estimates over H+A frames. Individual outputs are explicitly unverified tracklet proximity proxies with an unassigned bucket, not reliable player contributions. See PROJECT_PLAN.md for the team's chosen scope and AUTHORS.md for the coauthors.

## Intended use

Research on team possession segmentation from automatic tracking. The trained checkpoint is not a validated production model and is not evidence of qualification. No competition ground-truth scoring or video-output checks were performed.

## Original pilot: retained test result

External dataset: IDSSE/Sportec match J03WOH (Fortuna Düsseldorf vs SSV Jahn Regensburg). Source: Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, doi:10.1038/s41597-025-04505-y. Data licence: CC BY 4.0; DFL-authorised publication. Tracking and labels are from a different match from the competition.

One downloaded complete external match was available. Its first half is split chronologically at 70%, with 100 frames removed on each side of the split. The earlier portion is training, the later portion is validation, and the entire second half is the pilot test. The test was scored after settings were frozen. Same-match teams and conditions make this weaker than match-held-out evaluation. No confidence interval or generalisation claim is made.

| Method | Clean test spellF1 | Artificial ball-gap test spellF1 |
|---|---:|---:|
| Official smooth50 baseline | 0.3391 | 0.2385 |
| Direct learned classifier | 0.6000 | 0.4200 |
| Fixed-penalty sequence model | 0.6482 | 0.5530 |
| Possession Lens boundary verifier | 0.6489 | 0.5746 |

Most improvement over the official baseline comes from learned observation features and sequence decoding. The verifier's incremental gain is 0.0007 clean and 0.0216 under synthetic gaps. This does not establish that the verifier is reliably better.

Architecture: standardised multinomial logistic state classifier, independent binary boundary verifier, Viterbi decoder with observation-dependent transition costs. Features use ball proximity and movement, team geometry, player speeds, missingness and offline temporal context. No player names or club names are model inputs. State-classifier training includes team-swap augmentation. There is no Smart Tagging or video input in this pilot.

Training and inference stay separate. Labels only supervise training or external scoring and never enter observation feature computation. The original checkpoint uses feature version `geometry-context-v1` and has state/boundary weights, training means/scales and frozen decoder settings.

## Interaction and forecasting additions: validation only

These additions were made after the original test report was viewed. They were therefore developed and measured on the external training/validation partitions only; the original second-half test was not scored again.

Interaction evidence jointly uses 2D proximity, ball direction changes, relative velocity, motion alignment and closing movement. Implausible slot displacement is masked. Exported interaction candidates are **not confirmed contacts**. No touch-reference labels were available; contact-detection accuracy has not been measured. Aerial balls, track jumps and simultaneous challenges remain major limitations.

With the original validation-selected decoder settings held fixed, the interaction-enhanced boundary model scored 0.5398 clean and 0.4183 with gaps on validation. The corresponding interaction sequence model scored 0.5804 and 0.4500. The enhanced verifier did not improve this comparison and should not replace the simpler sequence model on this evidence.

The independent forecasting head predicts a direct live H↔A turnover within the next second before a stoppage. It excludes current dead/unknown frames, unknown reference horizons and incomplete end-of-partition horizons. Features use current/trailing observations only, never future features or offline detector outputs. A unit test changes all future observations and checks that earlier forecast features remain identical. Provider tracking may itself have been smoothed offline, so a live deployment claim is not justified.

| Forecast validation measure | Clean | Artificial gaps |
|---|---:|---:|
| Average precision | 0.2343 | 0.1778 |
| Prevalence reference | 0.0896 | 0.0896 |
| Brier score (lower is better) | 0.0834 | 0.0857 |
| Training-prior constant Brier | 0.0825 | 0.0825 |

The risk ranking exceeds the constant-score reference, but probability error is worse than the constant-probability baseline. Do not present these scores as calibrated operational risk. Event-level alarm precision, lead time and false alarms per minute have not been established.

## Limits and next evidence needed

- Only one complete external match is currently available locally. Other external downloads timed out.
- Artificial gaps remove about 40% of ball observations in random 0.8–4-second runs. They do not reproduce all real missingness or long occlusions.
- Player dropout, identity-switch stress, duplicate-track stress and noisy tags have not been evaluated.
- External possession definitions may not match the competition definition exactly.
- Next test: train on J03WOH, select settings on J03WPY, test on a previously unseen match. The CLI supports J03WMX as the initial independent match-held-out test.
- All competition training/tuning and output-versus-video checking remain prohibited under the organiser's latest clarification.

## Reduced-switching development run

The original recurrent weights plus an external-validation-selected offline Viterbi penalty of 6 improved clean validation spellF1 from 0.3408 to 0.5138 and masked validation spellF1 from 0.2787 to 0.4848. Clean predicted spells fell from 136 to 56 against 60 reference spells. Team-swap/dropout augmented training was also tested but not selected on mean clean/masked validation performance. The scorer-selected baseline achieved 0.2690 clean and 0.2156 masked. These remain single-match validation results; no fresh test has been scored. See RNN_IMPROVEMENT.md and models/rnn-decoded-001-report.json.

## Team-symmetric inference and fourth-match confirmation

The latest checkpoint is models/rnn-symmetric-001.npz. It averages original and team-swapped recurrent probabilities with H/A outputs realigned, followed by offline penalty-24 Viterbi decoding. Weights and training standardisation are unchanged. Selected on J03WPY validation, it scored 0.3707 clean / 0.2883 masked spell F1 on fresh J03WN1, versus the previous model 0.2884 / 0.2426 and baseline 0.2838 / 0.2083. This candidate was not scored on J03WMX after development. Separate stop penalties and transition-weighted training did not improve validation and were rejected. The full audit is in RNN_SYMMETRY.md and models/rnn-symmetric-001-report.json. Both external tests are now viewed; subsequent changes require fresh evidence.

## Boundary-aware recovery (11 November 2026)

Latest recommended checkpoint: models/rnn-boundary-001.npz. A training-only logistic boundary verifier adjusts transition costs around the frozen symmetric RNN. Selected on J03WPY, it improves J03WQQ clean spell F1 from 0.4860 to 0.5025, misses from 246 to 237 and false positives from 81 to 75. Dropout F1 is essentially unchanged. Clean merge-overlap count stays at 55; masked count falls 49 to 47. The duration-weighted alternative recovers more spells but adds false positives and is experimental. The new test shares a home club with training, score gains are modest and no significance/competition claim is made. All three test matches are now viewed. See RNN_RECOVERY.md for the full audit.
