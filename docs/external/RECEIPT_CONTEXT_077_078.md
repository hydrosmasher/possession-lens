# Receipt compatibility experiments 077–078

16 November 2026. **Decision: preserve frozen model053 and its submission CSV.** Both experiments are rejected for deployment. Six external-trained diagnostic scorers are archived separately; no production model/configuration changed.

## Question and changes

Can richer receipt evidence reject a ball passing close to an opponent while retaining genuine short possession spells? Experiment076 showed that suppressing existing short pulses can lose already-detected possessions. This study therefore filters **only novel070 recovery proposals**; it does not directly edit an existing053pulse.

077 compares two matched classifiers: original14 motion features versus34 features with longer same-slot continuity and competition evidence. The extra features measure ball/player relative velocity, approach/departure distance, same-player proximity persistence, nearby opponent distances and team-relative margins. All inputs are observed coordinates and team memberships; no player names, reference labels, match clock, or Barcelona prior enters features. Both models use exactly the same eligible training frames, so the comparison controls for the stricter long-context availability.

The current nearest player is followed through +/-24frames (0.96s at25Hz). Missing ball or same-player track observations invalidate the context. This is an offline ownership-compatibility scorer, not a verified touch/receipt detector or a live prediction claim. Possession during a pass can legitimately be assigned before any player touches it, which limits ownership labels as receipt supervision.

078 is a separately preregistered adaptive follow-up: same34features, learner, seed, splits and calibration; weight owning-team examples inside <=3s reference spells by4 and close-opponent negatives by2 during fitting. Other samples have weight1. Evaluation labels never choose these weights or the cutoff. No post-result cutoff sweep occurred.

## External-only protocol

Data: existing licensed IDSSE/Sportec WMX/WN1 first-division Bundesliga tracking and H/A/D reference, CC BY4.0. Opposite-match folds, both halves, clean/short-gap/long-gap/tracker-noise cases. These two matches and error draws are heavily reused development data, **not fresh independent tests**. Counts below sum repeated degradation cases, not independent matches/events.

For each fitting match: first70% of each half for model fitting, last30% for calibration, with a50frame embargo on either side of the split. Feature windows never cross the split. Known-live close candidates sampled every5frames;120 fixed HGB iterations,15 leaves, minimum100samples perleaf, learningrate.05,L2=20,seed177,noearlystopping. Models are exported to pickle-free numeric trees and checked against native probabilities.

Calibration uses genuine external reference short spells entirely in calibration regions. Each eligible spell contributes its peak score over the old064-supported frames. The cutoff is the minimum of those peaks, ensuring all eligible calibration short spells survive. No eligible groups means cutoff0 and no authority to filter. There were136/127eligible groups across the augmented calibration cases in the two folds; these include repeated views of the same physical spells.

Only accepted070proposals are reconsidered. A proposal is retained if any old-supported frame passes the new cutoff. If any old support lacks full long-context validity, the filter abstains and retains the proposal. Original D labels, missing-ball frames and frames outside novel proposals are asserted unchanged. Thresholds and model hashes were written before each opposite-match evaluation.

## Pipeline results

| Method | Mean spell F1 | Missed | False positives | Matched short spells |
|---|---:|---:|---:|---:|
| Frozen 053 | 0.628641 | 1603 | 1101 | 787 |
| Recovery070 | 0.631917 | 1561 | 1137 | 824 |
| 077 matched14 | 0.632060 | 1561 | 1135 | 824 |
| 077 context34 | 0.631917 | 1561 | 1137 | 824 |
| 078 weighted34 | 0.631917 | 1561 | 1137 | 824 |

The matched14branch rejected one proposed episode and reduced false-positive spells by2, with no loss of any short spell matched by070 and retention of all46newly recovered GTspell matches across the repeated cases. Its0.632060 mean exceeds070's0.631917, but FP1135 remains above053's1101. On WMX tracker noise it scores0.524973 versus053's0.526092. It also is the matched control, not the predeclared promotable new-feature branch.

The34feature model rejected no proposals at its retention-constrained cutoffs (0.175767/0.263226). The weighted follow-up also rejected none. They reproduce070's pipeline results and noisy-fold regression. Both preserve all824shortspell matches from070 and all46newrecoveries, but do not solve false-switch precision.

## Component findings and limits

On exactly the same eligible evaluation frames, the34feature model improves owning-team-vs-opponent ranking:

| Opposite evaluation match | 14feature ROC AUC | 34feature ROC AUC |
|---|---:|---:|
| WN1 | 0.816313 | 0.888912 |
| WMX | 0.863967 | 0.923463 |

These are **component ROC AUC**, not spell F1 or challenge accuracy. At the frozen retention-safe cutoffs, the richer model retains more owning candidates but also admits more opponent decoys (61.99% versus59.57% pooled). Better ranking alone is insufficient: at least one genuine calibration short spell has weak evidence, forcing a permissive cutoff, and proposal-level peak aggregation tolerates misleading isolated support.

The weighted follow-up failed to fix this operating-point issue. We did not raise thresholds against the opposite evaluation scores or discard the hard short examples. Exact matched-spell retention, false-positive counts and per-fold noisy-tracking checks all remain mandatory.

No different-error-draw promotion audit was run for rejected candidates. No supplied challenge match/video/labels/public score was used in training, calibration or selection. No new data was downloaded. Existing research, model053 and the submission bundle remain intact.

## Reproduce and review

Public review needs no private input: inspect the committed reports/protocols, archived numeric weights, manifest and synthetic tests. The compact judge notebook displays aggregate comparisons with inference off. The preserved research notebook adds safe-by-default optional replay cells.

For full research reproduction, use Python3.12 plus the gated/research dependencies (including scikit-learn1.7.2, torch2.9.1), licensed WMX/WN1 caches, and the preserved ignored053/064/069/070 research caches. Use a separate research checkout/new output folders; runners refuse to overwrite earlier results:

```sh
python -m controlswitch.receipt_context_experiment --root . --cache-root /path/to/licensed/external/cache
python -m controlswitch.receipt_weighted_experiment --root . --cache-root /path/to/licensed/external/cache
python -m unittest discover -s tests -p test_receipt_context.py -v
```

078 also requires the locally reproduced077run. Public archival model weights alone are not a complete replacement for the private generator caches. All detailed per-frame arrays stay ignored under local/.

Next useful experiment: calibrate an episode-level discriminator with external short-control boundary evidence and explicit opponent-decoy groups, rather than accepting any high-scoring ownership frame. Preserve uncertain cases and keep the same exact-retention guards. More layers or longer context alone has not solved the issue.
