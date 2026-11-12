# Recovering missed and merged possessions

This experiment continues the European-only RNN with a different reserved match. Training remains 4039 Manchester City–Liverpool, 3749 Dortmund–Bayern and 3518 Juventus–Inter. Validation remains 2269 PSG–Marseille. Previously tested 3442 Real Madrid–Barcelona is excluded from development. The new test is **2417 Bayern Munich–Borussia Dortmund, 9 November 2019, Bundesliga**. It shares both clubs with training; do not describe it as unseen-team generalisation.

The fresh file was extracted from the already downloaded official historical SkillCorner archive, pinned at `1ed50b775f93075c2a6d00d0d795f13b7b89709e`. Metadata and tracking file sizes and Git blob hashes were checked against the official GitHub directory before fitting. Tracking contents and possession labels remain unparsed until the candidate, decoder, baseline and comparators are frozen. MIT attribution is retained in `docs/licences/SkillCorner-MIT.txt`.

## Changes being tested

1. **Spell-balanced recurrent loss.** A new 24-unit, one-layer RNN is trained from scratch. Short, fully observed H/A spells receive more loss weight: `clip(sqrt(250 / spell_frames), 0.5, 4)` on the common 25 Hz timeline. A neighbourhood within 12 frames of a direct known H/A switch receives an additional 1.5 multiplier. Spells truncated by unknown intervals or half boundaries retain base weight 1, so incomplete observations are not mistaken for genuinely short possessions. Unknown frames have zero weight. This changes supervision, not inference inputs.
2. **Adaptive switching costs.** A separate logistic boundary head learns from team-invariant proximity, trajectory, movement and RNN-score changes in training matches only. Positive/negative boundary samples are balanced. Its output is an event score, not a calibrated natural-frequency probability. A 25-frame neighbourhood around unknown labels is excluded from boundary supervision. The head reduces transition cost where evidence supports a switch; emissions must still support the new team. There is no hard minimum duration beyond one frame.
3. **Validation safeguards.** Plain and adaptive decoders are compared with penalties 3, 12, 24 and 48, and boundary strengths 0, 0.5 and 1. A candidate is eligible only if its pooled clean/dropout false-positive spell count at IoU 0.5 is no more than 110% of the previous European RNN, and its sub-0.4-second pulse count is no more than five higher. Highest mean validation spell F1 wins; the unchanged previous model remains an explicit fallback. There is no target spell-count matching.

The recurrent run uses 12 sampled iterations, at most 1,024 windows per iteration, 128-frame windows, 16-window batches, dropout 0.15, L2 0.0001, Adam, gradient-norm clipping at 5 and seed 42. It compares all iterations by full validation spell F1, retaining loss probes to diagnose overfitting. An iteration is not a full epoch. The boundary heads are fitted for 35 epochs. Training data alone determines normalisation and gradients.

## Honest evaluation

All comparisons use identical tracking and provider-label masks. Labels are automatic and cover only observed H/A intervals; null stays X and is never converted to dead ball. Unknown gaps censor spells and can inflate F1. Interpolation to the common 25 Hz interface adds no independent evidence. These are conditional transfer scores, not independently annotated full-match ground truth or official challenge accuracy.

The fresh test is scored once after saving the checkpoint and source hashes in the freeze record. Clean and simulated ball-dropout conditions use test seed 400. Reports compare the candidate, previous European RNN, production model and validation-selected baseline, with missed spells, false positives, duration-bin recall, short pulses and geometric merge counts. A geometric merge means a predicted spell overlaps more than one same-team reference spell; it is descriptive, not a separately confirmed event.

No post-test tuning is allowed. Reused validation can itself overfit. A test improvement does not automatically promote an H/A-only candidate to the production model, whose dead-ball behaviour is not evaluated by these labels. The 0.8–0.9 target is an aspiration, never a guaranteed score.

## Reproduce

Open `notebooks/01_Possession_Lens.ipynb`. Default cells display reports and run checks without importing, training or challenge inference. Optional cells verify the sixth-match archive import and reproduce recovery training. Reproducing a viewed test is not fresh evaluation. The notebook records that distinction in its copied protocol.

The shareable checkpoint and report use the prefix `rnn-europe-recovery-001`. Raw assets, sealed import manifests and full local run history remain in ignored `data/` and `local/research/` folders. No challenge files are used by the experiment, and no competition output is compared with video.
