# Player control development

## Available now
Anonymous track proximity shares, unassigned time, auditable control intervals, dashboard CSV and private report exports. Shares use total predicted team possession, including unassigned time, as denominator. Tracks are not people or jerseys. Nearby ball flight can be associated; this is not contact ground truth.

## External identity experiment 002
Motion-only, bidirectional constant-velocity fragment linking; team consistency, no overlap and ambiguity rejection. PersonId is used only for external scoring. J03WPY first-half motion establishes a speed limit; second-half synthetic fragments select link gates. No provided game is used. Validation gate was precision >= 0.995 with at least 30 links. Best precision: 3570/3588 = 0.9949833. The gate failed and no model was approved. J03WN1 identity test was not run. Prior attempt 001 also failed validation; attempt 002 introduced bidirectional consistency and more conservative gates on validation only. J03WN1 has previously been viewed for the separate team-possession task, so it is not an entirely unseen match.

Simulation uses 6–30 second fragments and 1–25 missing frames. It lacks appearance errors, long occlusions and real identity switches. Do not round a failed result into a pass. Next work should improve reciprocal matching and test with more realistic corruption on external training/validation data before opening the reserved identity test. Never force eleven identities or hand-label the provided match.

## Reproduce locally
Use controlswitch.player_identity.calibrate(train_xml, train_cache, test_xml, test_cache, new_output_folder), with J03WPY and J03WN1 inputs. Failed validation raises ValueError and preserves protocol/trials. Successful validation freezes a candidate before the identity test. Current source must not link competition tracks on this failed evidence.

## Named external mapping
The provided game has only generic Name fields and sequential slot-like ShirtNumber fields, not a real roster mapping. No names are guessed. The named-external command demonstrates the complete mapping path on licensed J03WN1 data: verify match/team metadata, verify XML against its parsed cache hash, align PersonIds to tracking slots, join by team + PersonId to provider roster, aggregate identical people across halves and preserve unassigned time. Missing/duplicate/mismatched identities raise errors. The dashboard and CSV support named rows only in explicitly external examples. A real challenge mapping still requires a permitted source and trustworthy identity continuity; manual slot-to-person mapping on that game is disallowed.

Example reproduction:
```sh
python -m possession_lens named-external --match J03WN1 --tracking data/external/raw/DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml --metadata data/external/raw/DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml --cache data/external/cache/J03WN1-56ef948961a1.npz --model models/rnn-boundary-001.npz --out local/named-example-reproduction
```
