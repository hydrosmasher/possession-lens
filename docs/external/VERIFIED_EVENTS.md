# Verified event stats — Bochum–Leverkusen

The user requested verified events only. The nine SkillCorner tracking examples do not establish a complete pass/shot/touch event feed, so those event counters are **Not available**, never inferred or zero-filled.

The existing Sportec Bochum–Leverkusen example (J03WN1, 27 May 2023) now includes its openly downloadable CC BY 4.0 event XML, joined to the existing provider match roster by exact PersonId + team. File identity, both half boundaries, unique event IDs, actor names, team mappings and team/player totals are checked. Deleted provider events are excluded. No anonymous identity or guessed name is used.

Pinned source: `pysport/idsse-data`, revision `a715a38dfbaf5f58e431727c2b78d174101a703c`, filename `DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml`. The browser saved the XML as source text, and its Git blob SHA-1 was independently verified as `67b982007b8e3f5de83dcd104779acc92b76a813` (612,114 bytes), matching the source ETag. Raw XML is not bundled; transformed summaries retain attribution.

Definitions and counts:

| Metric | Bochum | Leverkusen | Definition |
|---|---:|---:|---|
| Completed passes | 271 | 253 | Provider Play with Pass or Cross, evaluated successfullyCompleted/successful; includes crosses and set pieces. |
| Shots | 10 | 11 | ShotAtGoal records. |
| Shots on target | 5 | 2 | SuccessfulShot or SavedShot; blocked and woodwork excluded. |
| Goals | 3 | 0 | SuccessfulShot records. |
| Touches | Not available | Not available | The feed does not establish every physical contact. Control bouts are not relabelled touches. |

The website has a separate provider event timeline for goals, substitutions, shots and completed passes. It uses wall times relative to each half kickoff. The model's tracking-time possession timeline remains separate. Half filters apply to event totals and the event timeline; player control shares remain full-match figures, as labelled. Verified per-player counts join to the same roster IDs and are included in player CSV exports; unavailable metrics are blank rather than zero. These are provider-defined counts, which may differ from another supplier's published totals.

Credit: Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, DOI https://doi.org/10.1038/s41597-025-04505-y; IDSSE/Sportec, authorised DFL data, CC BY 4.0. Dataset DOI https://doi.org/10.6084/m9.figshare.28196177. Derived file: `examples/verified-events-J03WN1.json`.

StatsBomb's public README/licence were also checked, but no StatsBomb event data were imported or redistributed. Its registration/reuse conditions were not accepted on the user's behalf; the anonymous-download constraint remains in force.

Rebuild after independently obtaining the licensed XML and metadata:

```sh
python -m controlswitch.verified_events \
  --events data/external/raw/DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml \
  --metadata data/external/raw/DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml \
  --out examples/verified-events-J03WN1.json
python -m controlswitch.example_gallery --raw data/external/skillcorner-europe/raw \
  --model models/rnn-europe-context-001-experimental.npz --out examples/matches
python -m possession_lens export-demo --out dashboard.html
```
