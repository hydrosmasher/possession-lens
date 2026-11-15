# Submission release status —15 November 2026

Starting benchmark: candidate 052 **0.615370**. Final checkpoint: **053, overall external development possession-spell F 1 0.628641**. The target is 0.85–0.90; it remains unmet, and competition rank is unknown.

The selected GRU continues 052 for 12 fixed iterations × 512 windows (6,144 additional sampled windows; 52 iterations and 26,624 sampled windows including its 043/052 history). Half of its training anchors are drawn uniformly from known-live spells of at most 3 seconds, then uniformly within that spell. Those frames receive twice their previous spell-weighted CE weight. AdamW learning rate 0.0001, weight decay 0.001, gradient clipping 1, seed 153, dropout 0.25, spatial reflections and team swaps remain fixed. The unchanged two-layer bidirectional 64-unit GRU still uses 508 features, projection 64, LayerNorm and the narrow auxiliary boundary objective. Forward weights fit WMX only; reverse development fits WN 1. Labels affect training sampling/loss only and never enter inference.

The two Bundesliga matches are heavily reused external development data. Four conditions pool opposite-match predictions: clean 0.727686, short ball gaps 0.636442, long gaps 0.610947, tracking noise 0.539490. These use the organisers' official spell metric, not the challenge ground truth. No challenge video, labels or public score is used for fitting, selection or comparison.

After the fixed winner was selected, different simulated-error draws scored 0.606253 → 0.622752. These are the same matches, not an independent-match test.

Prepared CSV: `Visca el Barça, visca Catalunya.csv`, SHA 256`2f5a445d2c3c10b3126b5bda3a5b9b11050d9a87410f19b7cbc5aaa88b928eeb`. Notebook uses registered Python 3.12 GRU kernel. Old models/bundles and evaluation/Submission.csv are preserved. Match inputs, feature caches, private outputs and licensed assets remain excluded from GitHub and source ZIP. Repository is private; public release and form submission are pending. No provided-game accuracy or competition percentile is established.
