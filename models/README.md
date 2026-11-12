# Frozen external-trained models

`rnn-pilot-001.npz` is the first 24-unit tanh recurrent model, trained only on the documented external IDSSE/Sportec development partition. Its report records the training data hashes, sequence split and validation-only results. No competition inputs were used to fit the checkpoint or its standardisation statistics.

Training data attribution: Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, doi:10.1038/s41597-025-04505-y. Dataset licence: CC BY 4.0, authorised by DFL. The checkpoint is a development artifact, not a verified winning model or an individual-player attribution model.

`rnn-decoded-001.npz` adds the external-validation-selected offline transition penalty (6). The recurrent architecture remains unchanged. Its companion report includes all plain/augmented training and decoder trials, the baseline comparison, source hashes and an explicit absence of fresh-test results.

Capacity experiments 001 (16 sampled iterations) and 002 (up to 32 with early stopping) retain one-, two- and three-layer checkpoints as rnn-capacity-00N-Dlayer-experimental.npz. These use the numpy-stacked-tanh-rnn-v1 loader and fixed symmetric penalty-24 decoding. The one-layer variant still gives the best validation spell F1. See docs/RNN_CAPACITY.md and reports/external/rnn-capacity-002-report.html. Experimental weights have not replaced rnn-boundary-001.npz.
