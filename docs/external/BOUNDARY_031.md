# Boundary alignment031

This experiment holds the frozen030 context RNN, interaction-memory team classifier and quality-gated dead-ball evidence fixed.020's turnover head was fitted with019 emissions;031 tests whether refitting its existing116 features on030 emissions improves segmentation. Training uses only the existing licensed first-division Bundesliga games, in opposite-match directions. These games have been extensively reused; they are development data, not fresh tests.

Before outcomes, the protocol fixes three candidates: the original head at full adaptation strength, the aligned head at half strength, and the aligned head at full strength. Training retains the existing known-live direct-turnover target and balanced logistic fitting. Model selection uses official spellF1, with a minimum0.003 mean gain and per-fold condition/dead-state tolerances. False-positive spells, misses, recoveries and merges are reported. A selected head is hashed before a different-observation-draw check; no runner-up selection follows that check.

The purpose is a better possession sequence, not a larger network for its own sake. No provided-game observations, labels, video or public score enter this experiment. All context is offline and can use future observations. Competition percentile is unknown.

Protocol and results are retained in reports/external/boundary-031-*.json. The notebook displays the completed report; optional refitting stays disabled by default.030 remains frozen until a new candidate meets all predefined checks.

## Completed result

| Condition | Frozen030 spell F1 | Candidate031 spell F1 |
|---|---:|---:|
| clean | 0.6053 | 0.6248 |
| masked | 0.5058 | 0.5284 |
| long_gap | 0.4842 | 0.4968 |
| tracker_noise | 0.4238 | 0.4388 |

Mean: **0.504776 → 0.522219**. Repeated-condition misses2307→2175, geometric merges522→482, false-positive spells918→1031.153 spell instances are newly recovered,21 previously recovered instances are lost. Counts repeat physical possessions across conditions and do not establish independent sample sizes.

Ablation: old head/full strength0.521253; aligned head/half strength0.504466; aligned head/full strength0.522219. Most of the improvement comes from stronger adaptive decoding; refitting alone does not improve the mean. The selected aligned/full head passes every predefined check. Forward head fits7,448positive and7,448negative sampled exposures from WMX only; reverse head fits7,264+7,264 from WN1. These include repeated conditions and are not unique events.

After freezing, a different observation draw improves **0.506851 → 0.519108** against030 on identical inputs. No alternative was selected using this audit. Candidate031 is recommended;030 remains reproducible via its unchanged config. There is no provided-match accuracy or competition-percentile claim.

The public031 config reuses the engine's ranked-possession-030 schema and changes its candidate identifier, boundary asset and full-strength specification. Inference never refits or calibrates. Exact saved-head/decoder replay, CSV hash and environment evidence are in the031 replay/reproduction reports.
