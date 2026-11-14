# Ranking-first candidate030

The user clarified the objective as top5% of competition entries, not95% frame accuracy. Official possession-spell F1 is the selection metric. No leaderboard distribution is available and no percentile is claimed.

## Method

028 adds ball/player relative movement, soft interaction evidence and short/long past/future team association to the244-feature control classifier, producing296 features. Associations are geometric evidence, not verified touches.029 learns27 team-invariant observation-quality features from automatically corrupted external tracking; it gates between the existing clean/ball-gap and tracking-noise dead experts in log-odds space.020's balanced boundary head then adapts the switch penalty at strength0.5. The original context RNN remains a component. All assets are numeric, pickle-free NPZ and hashed in the config.

## Selection history

028/029 were configured before the ranking-goal clarification. Their original no-regression guard rejected them, including a two-frame clean-accuracy difference. Before inspecting their outcome reports,030 recorded the clarified ranking-first policy. It compares ten fixed standalone/combined variants. Official mean spellF1 is primary, with per-condition/fold spell and dead-state regression tolerances. False-positive counts are reported rather than an absolute veto; this explicitly supersedes the earlier policy. The selected model and hashes were frozen before a new-observation-draw audit. No runner-up was selected using that audit.

## Results

| Reused external condition |019 spell F1 |030 spell F1 |
|---|---:|---:|
| clean | 0.5729 | 0.6053 |
| masked | 0.4720 | 0.5058 |
| long_gap | 0.4672 | 0.4842 |
| tracker_noise | 0.3382 | 0.4238 |

Four-condition means: **0.462596 → 0.504776**. Pooled repeated-condition missed spells2458→2307, false-positive spells1100→918, geometric merges544→522. Newly recovered233, lost82. These counts repeat physical possessions across simulated conditions. Individual conditions can have more false positives: full per-fold diagnostics are published, not suppressed.

After freezing, different augmentation seeds on the same matches give **0.458769 → 0.490035**. This is a robustness check, not a fresh-match test. Both Bundesliga matches have been extensively reused; provider labels/tracking are correlated. Synthetic tracking errors may not represent the challenge domain. No provided-match training, video comparison or public-score selection occurred.

## Reproduction

Use README's ranked_pipeline command and reports/external/ranking-030-config.json. Saved weights exactly replay all eight forward external predictions. Two source/dependency environments reproduce CSV SHA256 `da4bb49ef582071904cb00d0a04c010eea50d2ae7124a24c601499f83e7b807c`. No challenge accuracy or competition percentile is measured. Previous019 remains reproducible through its original config/assets and command.

Reports: ranking-030-protocol.json, ranking-030-report.json, ranking-030-freeze.json, ranking-030-replay.json, ranking-030-reproduction.json. Accuracy027 is diagnostic;95% frame accuracy is not the clarified goal. Native028/029 reports retain their original guards for a complete research history.
