# Reproducing Possession Lens

## Current submission: candidate053

Use Python3.12. The root README contains the exact inference command. Install `requirements-gru-inference-lock.txt` for inference or `requirements-gru-notebook.txt` for inference plus JupyterLab. Earlier tanh-RNN commands are research history, not the current submission path.

The data-root folder must contain one copy of the tactical tracking XML and TXT. The importer searches recursively and rejects duplicate matching filenames. Obtain assets from the organisers; neither raw data nor local symlink manifests are distributed with the source.

```sh
python -m controlswitch.recurrent_pipeline --data-root /path/to/provided-data --out local/judge-reproduction --config reports/external/short-spell-053-config.json --filename "Visca el Barça, visca Catalunya.csv"
python evaluation/official/evaluation/validate_submission.py "local/judge-reproduction/Visca el Barça, visca Catalunya.csv"
```

Use a new output directory: inference refuses to overwrite earlier runs. No training or network service is used during inference. The config verifies SHA-256 values of nested configuration and model assets before loading. Pickle-free model files are used. CPU inference uses one PyTorch thread and deterministic algorithms.

Expected CSV SHA-256: `2f5a445d2c3c10b3126b5bda3a5b9b11050d9a87410f19b7cbc5aaa88b928eeb`. Latest-code reproduction on16 November2026 matched the frozen bundle exactly in the pinned Python3.12 environment. Cross-platform bitwise identity is not claimed; if a different platform differs, retain its manifest and investigate rather than editing labels.

## Outputs

- Team-named CSV: the only scored prediction file.
- `manifest.json`: input/model/config hashes, versions and inference policy.
- `team-statistics.json` and `timeline.json`: model estimates.
- `observation-quality.json`: missing-ball, limited-tracking and implausible-step flags within analysed halves. These do not change labels or provide calibrated confidence.

## Notebook

```sh
python -m pip install -r requirements-gru-notebook.txt
python -m ipykernel install --user --name possession-lens-gru-312 --display-name "Possession Lens (Python3.12, GRU)"
python -m jupyterlab notebooks/01_Possession_Lens.ipynb
```

Select that Python3.12 kernel. The notebook begins with the current deployed status and latest research outcomes. Optional downloads, training, imports and challenge inference stay disabled. Saved cell outputs are cleared. The evaluation notebook is separate and can read a CSV placed beside it; evaluation must not be used for tuning on challenge truth.

## Tests and research

```sh
python -m unittest discover -s tests -v
```

The full suite needs the pinned PyTorch environment. GitHub Actions installs CPU PyTorch before the same pinned inference dependencies. Private research-lineage tests skip when caches are absent. Research fitting additionally needs the declared licensed external inputs and optional scikit-learn; these are not required to reproduce the submission. Rejected research filters require explicit research opt-in and are not loaded by default inference.

The latest episode pilot uses provider-event phase windows, not matched Metrica automatic tags. Its counts are not evidence of challenge improvement. Earlier documentation is preserved in `docs/archive/` and numbered reports.

## Sharing and submission

The registered team is **Visca el Barça, visca Catalunya**. Submit the team-named CSV, a public repository URL and the team-named short technical document. The repository must not contain licensed challenge assets or reconstructive tracking exports. Use https://github.com/hydrosmasher/possession-lens and verify it opens without signing in before sending the form. Official deadline:17 November2026 at23:59CET, as stated on the submission form.
