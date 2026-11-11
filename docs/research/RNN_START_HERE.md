# Team possession RNN: first project milestone

Authors: Hrishikesh Karmalkar, Basel, Omar, Michail and Paul.

## Location

Project: `/Users/hydrosmasher/Desktop/Possession Lens`

Organised private assets: `data/raw/provided/tactical/` and `data/raw/provided/broadcast/` inside the project. Nine local file links point to the original Desktop downloads. No original files were moved or changed.

Trained checkpoint: `models/rnn-pilot-001.npz`.

First inference run: `work/rnn/predictions-001/`, containing the full-frame `possession.csv`, team-statistics JSON, provisional tracklet-proxy JSON, and a local timeline. These are experimental estimates and are excluded from the shareable source package.

## What was built and checked

- Actual 24-unit tanh recurrent neural network with backpropagation through time, implemented in Python/NumPy.
- Five training epochs on the existing external Sportec match cache; no fitting on the competition match.
- External validation spellF1: 0.3408. This is a starter run, not an optimised model or a competition score.
- Frozen checkpoint applied to the supplied tactical tracking. The 199,319-row CSV passed the official format validator.
- Nineteen tests passed, including numerical gradient checks, recurrent-state continuity, percentage denominators and unassigned missing-ball time.
- Pinned dependencies, GitHub CI configuration, author list and reproduction instructions are included. Nothing has been pushed to a remote GitHub repository.

## Recreate locally

From the project folder, install requirements as described in README.md. Judges can use the supplied checkpoint without retraining:

```sh
python -m possession_lens organise-data --source "/path/to/organiser-downloads"
python -m possession_lens rnn-predict --data-root data/raw/provided \
  --model models/rnn-pilot-001.npz --out work/rnn/judge-run-001
```

Training reproduction and the external-data downloader are documented in README.md. Raw external training XML is no longer in Downloads; the initial run used the previously parsed, source-hashed external cache. Source code, frozen weights and their report are packaged; cached/raw match data is not.

## Individual percentages

The input slots are not reliable individual identities. `tracklet-proxies.json` reports a geometric proximity estimate, with unassigned time retained. It must not be presented as per-player possession, jersey-number attribution or an RNN contribution explanation. Video availability alone does not resolve this. Named-player analytics and attack-versus-recycling usage are deferred until their prerequisites can be validated.

Latest competition rules apply: external-only training and tuning, no output-versus-video checks, no manual annotations of the provided match, and no uploads of restricted inputs to AI services.
