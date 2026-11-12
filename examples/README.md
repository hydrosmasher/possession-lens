# Named external player example

This is VfL Bochum–Bayer Leverkusen (J03WN1), not the challenge match. Names and shirt numbers are joined automatically using provider PersonId + team across tracking and roster XMLs. No challenge slots are mapped or renamed. Proximity shares use the frozen boundary RNN's predicted team possession, preserve unassigned time and do not establish physical ball contact or player-possession accuracy.

Source: Bassek et al. (2025), An integrated dataset of spatiotemporal and event data in elite soccer, https://doi.org/10.1038/s41597-025-04505-y. Dataset CC BY 4.0, https://creativecommons.org/licenses/by/4.0/; DFL-authorised publication. The files here are transformed summaries, not original tracking/possession labels. Full source and model SHA-256 values are in the analysis JSON. Raw external XMLs remain separately downloaded. No competition inputs are included.

Open external-named-dashboard.html or click Named-player example in the main dashboard. The names demonstrate the mapping pipeline, not a mapping for the challenge match.

## Ten-match offline library

`match-library.json` bundles nine historical SkillCorner top-five-league matches and the existing named Bochum–Leverkusen example. `matches/*.json` contains the nine individual SkillCorner summaries. These are transformed external summaries, not raw coordinates or organiser-match data.

The SkillCorner examples use `models/rnn-europe-context-001-experimental.npz` for estimated H/A possession. Frames with no observed ball or player are X, based on tracking availability rather than reference possession labels. The model does not classify dead ball, which the UI marks as “Not classified”. The examples include training, validation and previously viewed matches; only the separately frozen test report supports the stated 0.6286 F1. Browsing a gallery match is not another independent evaluation.

Provider `trackable_object` joins each named slot to the match roster. Anonymous nearest tracks and missing observations stay unresolved; they are never reassigned to a convenient named player. Named control seconds plus unresolved seconds reconcile to each team's predicted possession. Percentages retain the total team-possession denominator. Names are provider mappings, not manually verified identities. Impact means estimated proximity involvement, not confirmed touches, passes, goals, attacking quality or causal value. The displayed involvement count counts associated proximity bouts.

SkillCorner source: official historical `SkillCorner/opendata`, revision `1ed50b775f93075c2a6d00d0d795f13b7b89709e`, MIT; preserve `docs/licences/SkillCorner-MIT.txt`. The existing Sportec example retains its Bassek et al. (2025), DOI 10.1038/s41597-025-04505-y, CC BY 4.0 attribution.

Rebuild locally:

```sh
python -m controlswitch.example_gallery --raw data/external/skillcorner-europe/raw \
  --model models/rnn-europe-context-001-experimental.npz --out examples/matches
python -m possession_lens export-demo --out dashboard.html
```

The challenge match has no permitted stable-name mapping. Its player rows are hidden rather than displaying anonymous or invented player names; team-level local analysis remains available. Raw inputs and private outputs stay outside the source package.

Cologne–Bayern (J03WMX) has local tracking but no locally available verified roster. It is recorded in the library's unavailable-example inventory and withheld from the named gallery. A public roster download was attempted but not completed. No names were guessed. Other local second-division matches are outside the user's top-five-league scope.
