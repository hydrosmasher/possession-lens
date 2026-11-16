# Noisy tracking and possession fragmentation —079–082

16 November2026. **Decision: keep model053 and its exact submission CSV.** No candidate passes every predeclared check. Research results below are not challenge scores.

## What was tested

All studies use only the existing licensed CC BY4.0 IDSSE/Sportec first-division Bundesliga WMX/WN1 references and observations, with opposite-match frozen experts and four tracking conditions. These two matches and corruption draws are heavily reused development data, not independent tests. No challenge match data, video, labels or public score was used. No new data was downloaded and no new model weights were fitted this turn.

**079—duplicate and jitter stability.** Detect persistent same-team near-identical trajectories using at least250co-visibleframes,>=95%visibility-mask overlap and95th-percentile separation<=0.5m. Retain the most visible/smoothest track. Two alternatives recheck the local expert after duplicate removal alone, or duplicate removal plus a three-frame median where all neighbours exist. The detector uses observations, not a synthetic-condition flag. It found two duplicates in each noisy case and none in the other cases. Neither alternative vetoed a recovery: even cleaned observations left strong local support for both the true short receipt and the troublesome fragment.

**080—receipt-aligned prior agreement.** On duplicate-affected observations only, require053's unsmoothed team prior to favour the candidate at an old receipt-support frame, with fixed timing tolerances0or3frames. Both alternatives leave predictions unchanged. Independent-looking expert agreement therefore did not diagnose this error either. No cutoff was raised after evaluation.

**081—unsupported-gap fragment guard.** Revisit only novel recoveries. If an existing053same-team spell is within a1..75frame opposite-labelled bridge, at least half the bridge has missing ball and no old064opponent receipt support exists there, suppress the new isolated pulse. Original missing-frame labels are preserved; this is not gap interpolation or a verified receipt claim. The fixed75frame bound reuses the existing3sproposal bound, and majority means50%; no threshold sweep. One masked-ball proposal was rejected, reducing false-positive spells by2 without losing a recovered spell. This is a useful partial development improvement, **not a fix for the remaining noisy-tracking regression**.

**082—visible-bridge reattachment.** Starting from081, consider joining a recovered fragment to existing same-team control across at most75fullyobservedframes. Use a fixed equal blend of conditional053and069probabilities; score the bridge's log-likelihood gain plus the two removed boundary costs, using the existing adaptive12*clip(2-2*event,.25,2) formula. Choose at most one positive-gain side per proposal. No dead-ball crossing or missing-frame changes; only explicit bridge frames may change. This reconnects evidence without adding boundaries, but it loses four genuine recovered short spells. It is rejected, and no hand-picked subset of its joins is kept.

All rules, alternatives and acceptance conditions were written before each study's scoring. Later studies are explicitly adaptive follow-ups, not independent confirmation.

## Results

| Method | Mean spell F1 | Missed | False positives | Matched <=3s spells |
|---|---:|---:|---:|---:|
| Frozen053 | 0.628641 | 1603 | 1101 | 787 |
| 077 matched receipt candidate | 0.632060 | 1561 | 1135 | 824 |
| tracking-stability-079: dedup_consensus | 0.632060 | 1561 | 1135 | 824 |
| tracking-stability-079: dedup_median_consensus | 0.632060 | 1561 | 1135 | 824 |
| noise-agreement-080: exact_prior | 0.632060 | 1561 | 1135 | 824 |
| noise-agreement-080: tolerant_prior | 0.632060 | 1561 | 1135 | 824 |
| gap-fragment-081: gap_fragment | 0.632230 | 1561 | 1133 | 824 |
| visible-bridge-082: visible_bridge | 0.631857 | 1565 | 1133 | 820 |

| Method | WN1 tracker-noise F1 | WMX tracker-noise F1 |
|---|---:|---:|
| Frozen053 | 0.552804 | 0.526092 |
| tracking-stability-079: dedup_consensus | 0.556705 | 0.524973 |
| tracking-stability-079: dedup_median_consensus | 0.556705 | 0.524973 |
| noise-agreement-080: exact_prior | 0.556705 | 0.524973 |
| noise-agreement-080: tolerant_prior | 0.556705 | 0.524973 |
| gap-fragment-081: gap_fragment | 0.556705 | 0.524973 |
| visible-bridge-082: visible_bridge | 0.556705 | 0.524973 |

The best retention-safe partial result is081:0.632230mean spell F1,1561misses and1133false positives versus077's0.632060/1561/1135. All824shortspell matches from070 and all46newlyrecoveredmatches survive. Counts sum repeated degradation cases and are not independent real-world event counts.

However,053has1101false positives and the WMX noisy fold still regresses from0.526092to0.524973. Therefore081fails both the baseline false-positive guard and per-fold noise guard. 082additionally loses4of46recoveredmatches, leaving42, and drops short matches824to820. Model053is not replaced. No different-error-draw promotion audit is run for rejected candidates.

## Major diagnostic finding

The surviving WMX noisy recovery spans19frames of the correct team within a172frame reference possession. Its IoU with that possession is only19/172=0.110465, below the0.5spell threshold. It is a fragment of longer control, not a player falsely receiving the ball. Stronger receipt confidence alone cannot resolve its wrong boundaries.

The WN1comparison recovery spans15frames of a20frame genuine short possession (IoU0.75). Suppressing all small pulses would lose precisely the recoveries we need. This attribution uses automatic licensed external reference labels only; it is not hand annotation or inspection of the provided game/video.

The remaining WMX fragment's preceding69frame bridge is visible and has opponent receipt-proxy support, so the missing-ball guard correctly abstains. It should not be forced to veto via a threshold chosen for this one example. Visible-bridge joining likewise cannot be accepted after it erases genuine short possessions elsewhere.

Next useful direction: train and calibrate a contextual boundary/continuation discriminator on external examples of whole possession fragments versus real short control. Treat correct-team fragment labels differently from false receipts, and retain episode-level matching checks. None of this establishes a fresh-test improvement or competition ranking.

## Review and reproduction

The compact judge notebook shows the current model and one best research row per study; full alternatives and exact retention checks remain in the research notebook and committed JSON reports. Production inference remains unchanged and off by default in the notebook.

Synthetic checks run without private tracking:

```sh
python -m unittest discover -s tests -p test_tracking_stability.py -v
python -m unittest discover -s tests -p test_noise_agreement.py -v
python -m unittest discover -s tests -p test_gap_fragment.py -v
python -m unittest discover -s tests -p test_visible_bridge.py -v
```

Full study reproduction requires the Python3.12 research environment, licensed WMX/WN1cache and ignored local053/055/064/069/070/077research caches. Run079then080/081, and081before082in a separate research checkout. Runners refuse to overwrite existing results:

```sh
python -m controlswitch.tracking_stability_experiment --root . --cache-root /path/to/licensed/cache
python -m controlswitch.noise_agreement_experiment --root . --cache-root /path/to/licensed/cache
python -m controlswitch.gap_fragment_experiment --root . --cache-root /path/to/licensed/cache
python -m controlswitch.visible_bridge_experiment --root . --cache-root /path/to/licensed/cache
```

Public source and reports support review; the historical private research caches are not bundled with the repository. Raw per-frame arrays remain ignored underlocal/. Production053reproduction continues to use the documented public inference assets plus locally supplied licensed organiser inputs.
