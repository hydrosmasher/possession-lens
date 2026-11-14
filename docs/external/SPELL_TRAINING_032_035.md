# Spell-aware training032 and continuing experiments

032 keeps the RNN, dead-ball experts, quality gate and031 boundary head fixed. The296-feature control classifier adds tracking-noise examples and compares ordinary frame weighting with bounded inverse-square-root training-spell weighting. The median duration is computed within each training half, raw weights are clipped[0.5,3], then normalised to unit live-frame mean; final normalised weights can exceed3. Unknown/dead labels receive no control loss. Team swapping preserves weights.

Forward training uses136,592 repeated augmented live examples fromWMX, with100trees/15leaves/lr.08/minleaf40/L2=10/seed132. Reverse training uses109,408 examples. No new match was downloaded or called fresh. The official four-condition spellF1 and guards were fixed before outcomes.

| Reused external condition |031 spell F1 |032 spell F1 |
|---|---:|---:|
| clean | 0.6248 | 0.6434 |
| masked | 0.5284 | 0.5325 |
| long_gap | 0.4968 | 0.5104 |
| tracker_noise | 0.4388 | 0.4585 |

Mean0.522219→0.536213; repeated-condition misses2175→2110, false positives1031→1019, geometric merges482→472. Newly recovered163, lost98. Uniform noise training scores0.507568 and fails guards; the gain requires the changed loss weighting, not simply more augmented frames.

After freezing, different observation draws on the same games score0.513297→0.532568. This is a robustness audit, not independent-match testing.032 is the current candidate. Exact CSV reproduction is documented in spell-training-032-reproduction.json.

033 independently tests nonlinear116-feature boundary evidence and a softer switch-cost floor against031. Linear-soft scores0.522635; tree-default0.523136; tree-soft0.523817. None meets the0.003 minimum gain and all-condition guards. No033 audit or deployment was performed. Full rejected trials are retained.

The duration audit uses reference-match IoU0.5 recall, distinct from official multi-threshold spellF1. Clean031 recall is0.168 for0.4–1second spells versus0.878 for10–30seconds. Repeated physical spells across conditions are not independent samples.

035 is an optional, completed two-layer bidirectional GRU experiment with spell-weighted CE, auxiliary boundary loss, dropout, LayerNorm, AdamW, gradient clipping and fixed finite-context inference. Its25%blend scores0.541101 versus0320.536213, but fails a noisy-tracking fold guard;50%blend0.537655 and pureGRU0.495072 also fail. No035 candidate or audit was selected. The larger recurrent network is not deployed; all rejected trials and learning traces are retained. The current submission needs no PyTorch.

The two Bundesliga matches are heavily reused. No challenge video, labels, public score or observations enter training/selection. No0.7 or competition-percentile claim is supported. All previous releases remain preserved.

036 freezes those same GRUs and gates their influence with029 quality scores. Its25%cap scores0.537931, passing condition guards but failing the predeclared0.003minimum gain;50%cap scores0.530087 and fails. No036 audit or promotion.032 remains the recommended submission.034 provider compatibility and037 relative-motion feature experiments are still running and do not change the release.
