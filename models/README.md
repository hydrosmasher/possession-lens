# Frozen external-trained models

`rnn-pilot-001.npz` is the first 24-unit tanh recurrent model, trained only on the documented external IDSSE/Sportec development partition. Its report records the training data hashes, sequence split and validation-only results. No competition inputs were used to fit the checkpoint or its standardisation statistics.

Training data attribution: Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, doi:10.1038/s41597-025-04505-y. Dataset licence: CC BY 4.0, authorised by DFL. The checkpoint is a development artifact, not a verified winning model or an individual-player attribution model.

`rnn-decoded-001.npz` adds the external-validation-selected offline transition penalty (6). The recurrent architecture remains unchanged. Its companion report includes all plain/augmented training and decoder trials, the baseline comparison, source hashes and an explicit absence of fresh-test results.

Capacity experiments 001 (16 sampled iterations) and 002 (up to 32 with early stopping) retain one-, two- and three-layer checkpoints as rnn-capacity-00N-Dlayer-experimental.npz. These use the numpy-stacked-tanh-rnn-v1 loader and fixed symmetric penalty-24 decoding. The one-layer variant still gives the best validation spell F1. See docs/RNN_CAPACITY.md and reports/external/rnn-capacity-002-report.html. Experimental weights have not replaced rnn-boundary-001.npz.

## Expanded-data context RNN (12 November 2026)

`rnn-europe-context-001-experimental.npz` is the recommended **experimental European-transfer checkpoint**, selected on PSG–Marseille validation before the fresh Barcelona–Real Madrid (18 December 2019) test. It has one layer of 24 units, 120 multiscale tracking features, selected iteration 7 and decoder penalty 48. It is an offline model using future tracking context. Load through `controlswitch.rnn_runtime.load_frozen` and `infer_frozen`; do not feed the legacy 30-feature matrix directly to its weights.

The new model was trained from scratch on five top-five-league matches, with no provided-match input or second-division pretrained weights. On the same fresh test, mean clean/dropout spell F1 was 0.6286 versus 0.5804 for the previous European RNN. Clean misses decreased 125→115, false-positive spells 49→37 and geometric merge overlaps 29→27. Dropout misses/false positives also decreased; dropout merge count was unchanged. Partial provider labels support H/A only and cannot validate full-match dead-ball behaviour, so the production default remains `rnn-boundary-001.npz`.

The matching validation-selected ablations are in `archive/rnn-europe-context-001-instantaneous.npz` and `archive/rnn-europe-context-001-context.npz`. The primary candidate was fixed before scoring either ablation on the test. Full protocol, source hashes, learning history and diagnostics are under `reports/external/rnn-europe-context-001-*` and `docs/external/EUROPE_CONTEXT.md`. The 0.8–0.9 target remains unmet.
