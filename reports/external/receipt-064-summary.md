# Receipt compatibility: external experiment 064

The same nearby player is followed across a 0.96-second observation window centred on the candidate contact. Motion features include departure distance, relative speed, ball deceleration and direction change. The scorer distinguishes close-ball ownership compatibility; available labels do not establish individual receipts or touches.

| Check | Distance: owning candidate retained | Motion: owning candidate retained | Distance: opponent decoy accepted | Motion: opponent decoy accepted |
|---|---:|---:|---:|---:|
| Original external comparison | 94.83% | 94.26% | 84.17% | 58.32% |
| Frozen different-error draws | 95.03% | 94.23% | 83.94% | 57.24% |
| Existing different-provider match | 97.98% | 93.60% | 82.00% | 18.00% |

The original comparison reduces decoy acceptance by 30.7% relative to distance alone. The frozen error-draw check also passes the predeclared component gate. The provider-transfer check reduces decoy acceptance substantially but loses 4.38 percentage points of owning-candidate retention. It does not establish safe transfer.

The two Bundesliga games are heavily reused development matches. Training and calibration use disjoint temporal segments within the fitting match, with an embargo; the opposite game supplies evaluation. Different noise draws are not new matches. SkillCorner2440 was excluded from this component's fitting/calibration but used elsewhere in the project. Its check uses exact observed provider owner IDs and 5,398 eligible candidate points, including only350 opponent negatives. Native10Hz coordinates are resampled with the existing loader between adjacent observed samples;5Hz target sampling coincides with native observations. No identity recovery, manual labels or challenge data were used.

The conservative short-pulse filter changed no predictions: spell F1 remains0.628641, misses1603 and false-positive spells1101 on the repeated-condition benchmark. This is component progress, not a possession-score gain. No new submission CSV was generated;053 remains deployed and052 preserved.

Four new behaviour tests cover stopping versus flying past at identical distance, missing-ball abstention, label exclusion, pitch/team symmetry and preserving supported short interceptions.226 tests pass. Numerical tree export agrees with fitted classifiers to1e-12 on external calibration points.

Research-only exported models and thresholds are in `models/archive/receipt064-manifest.json`. Optional study commands, from the project root with the existing private research lineage/data available:

```bash
python -m controlswitch.receipt_experiment --cache-root data/external/cache
python -m controlswitch.receipt_audit --cache-root data/external/cache --raw-root data/external/skillcorner-europe/raw
```

Commands refuse to overwrite existing output directories. The research experiment additionally needs the frozen053 private fold-prediction caches, which are excluded fromGitHub; the default judge notebook reads shareable reports and weights without rerunning training. This experimental study is separate from the reproducible submission inference path.

Next issue: give receipt evidence a useful, bounded role in handover decisions without importing its cross-provider false negatives or changing unsupported frames.
