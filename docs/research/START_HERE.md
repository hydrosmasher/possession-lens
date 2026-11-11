# Possession Lens v0.2

## Open the website now

Double-click `control-switch-demo.html`. It contains an entirely synthetic example and requires no installation. Use the variant selector and timeline controls to explore the baseline. You can also select a full-length submission CSV: it is parsed in your browser without an upload.

## Project location

The Python project is `/Users/hydrosmasher/Desktop/Possession Lens`.
The four generated baseline CSVs are in `work/runs/baseline-001/` inside that project.
The Desktop dataset was read in place and was not changed or copied into the source.

## Run the website with actual local results

After following the dependency setup in the project's README, run from the project folder:

```sh
python -m possession_lens serve --run-dir work/runs/baseline-001
```

Open http://127.0.0.1:8765. No server is currently running from the build session: this environment blocked listening on a local port.

## Reproduce the outputs

```sh
python -m possession_lens baseline \
  --data-root "/Users/hydrosmasher/Desktop/MORE THAN A HACK" \
  --out work/runs/baseline-002
```

The output directory must be empty. Read README.md for the portable judge instructions.

## Verified

- All 199,319 tactical tracking IDs were validated in order.
- Four full-length baseline CSVs passed the official format validator.
- All four files were independently checked byte-for-byte against the original official baseline algorithm.
- Six Python ingestion and segmentation tests passed.
- Website code-level checks passed for synthetic initialization, variant selection, navigation, invalid CSV rejection and full-length CSV import without network requests.

## Limits

Visual browser QA could not be completed: the environment blocked the local server and the browser disallowed local-file URLs. Open the standalone demo locally to inspect its layout.

The original baseline remains available. Trained state and boundary models now have an external pilot report: open `external-pilot-results.html`. The clean test spellF1 was 0.6489 for Possession Lens versus 0.3391 for smooth50, but this was a second-half holdout of just one external match. It does not establish cross-match transfer or competition accuracy.

Open `interaction-forecast-results.html` for the later validation-only interaction and turnover-risk experiment. Those additions were not re-tested on the previously viewed test half. The interaction verifier did not improve on its simpler sequence comparator. Risk ranking exceeded a constant reference, but Brier probability error was worse; forecasting is not ready to present as calibrated risk.

See `MODEL_CARD.md` in the source package for the exact model, metrics and limitations. All training and tuning used external data. No predictions were checked against competition video or ground-truth possession. Tagging and homography are not yet model inputs.

`control-switch-source.zip` includes the source, upstream attribution, tests, model card, trained external checkpoints, reports and synthetic demo. It deliberately excludes match inputs, generated competition CSVs, match timelines and input-path manifests. Judges need their own authorised match data to reproduce actual competition outputs.
