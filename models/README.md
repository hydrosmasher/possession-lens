# Frozen external-trained models

`rnn-pilot-001.npz` is the first 24-unit tanh recurrent model, trained only on the documented external IDSSE/Sportec development partition. Its report records the training data hashes, sequence split and validation-only results. No competition inputs were used to fit the checkpoint or its standardisation statistics.

Training data attribution: Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, doi:10.1038/s41597-025-04505-y. Dataset licence: CC BY 4.0, authorised by DFL. The checkpoint is a development artifact, not a verified winning model or an individual-player attribution model.

`rnn-decoded-001.npz` adds the external-validation-selected offline transition penalty (6). The recurrent architecture remains unchanged. Its companion report includes all plain/augmented training and decoder trials, the baseline comparison, source hashes and an explicit absence of fresh-test results.
