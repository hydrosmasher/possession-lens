# Possession Lens — current recovery status, 11 November 2026

Recommended checkpoint: models/rnn-boundary-001.npz. Notebook defaults to it. Report: reports/external/rnn-boundary-001-report.html. Explanation and exact reproduction: RNN_RECOVERY.md.

Fresh external match: Fortuna Düsseldorf–FC St. Pauli (J03WQQ), 142,345 frames, 401 reference possession spells. Candidates, hashes, reference model, baseline and dropout seed were declared before scoring. Training/fitting used only J03WOH, settings used only J03WPY. Previously viewed J03WMX/J03WN1 were not loaded for development.

| Method | Clean spell F1 | Ball-dropout spell F1 |
|---|---:|---:|
| Baseline smooth50 | 0.3608 | 0.2254 |
| Previous symmetric RNN | 0.4860 | 0.4232 |
| Recommended boundary-aware RNN | 0.5025 | 0.4234 |
| Experimental spell-balanced RNN | 0.5055 | 0.4122 |

Recommended model: clean missed spells 246 → 237, false positives 81 → 75, no predictions shorter than 0.4 seconds. Clean merge-overlap count remains 55. Under dropout, missed spells 265 → 264, false positives 118 → 112, merge-overlap count 49 → 47. These are modest gains; merging is not solved and no statistical-significance claim is made.

The duration-weighted challenger recovers more spells but raises false positives and worsens dropout F1, so it remains experimental. Minimum-duration floors were tested but the winning settings use 1 frame (no added hard floor). All decoding and centred boundary context are offline.

All 38 tests pass. Frozen recommended inference completed and passed the official CSV format validator. No competition ground-truth/video check, training or tuning occurred. Source archive/GitHub exclude data and private runs.

All three test matches (J03WMX, J03WN1, J03WQQ) are now viewed. Subsequent development needs another independent match; J03WQQ also shares its home club with training, so this is not unseen-team evidence. No background evaluation remains running.
