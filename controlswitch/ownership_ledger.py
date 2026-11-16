"""Research-only ownership handovers; preserve the frozen dead-ball mask."""
import json
from pathlib import Path
import numpy as np
import torch
from .actor_association import BASE_CONFIG, actor_logits
from .player_set_rnn import PlayerSetPossession
from .pipeline import digest


def load_actor(path, expected_sha256):
    path = Path(path)
    if digest(path) != expected_sha256:
        raise ValueError('Actor checkpoint hash mismatch')
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['meta']))
        if (meta.get('version') != 'provider-actor-encoder-v1' or
                meta.get('nodes_per_team') != 16 or meta.get('node_features') != 21):
            raise ValueError('Unsupported actor checkpoint')
        model = PlayerSetPossession(BASE_CONFIG.copy(), pooling='global')
        weights = model.state_dict()
        expected = {k for k in weights if k.startswith(('encoder.', 'attention.'))}
        if set(archive.files) != expected | {'meta'}:
            raise ValueError('Unexpected actor weights')
        for key in expected:
            value = archive[key]
            if value.shape != tuple(weights[key].shape) or not np.isfinite(value).all():
                raise ValueError('Invalid actor weights')
            weights[key] = torch.from_numpy(value.copy())
        model.load_state_dict(weights)
    return model.eval()


def association(nodes, model=None):
    """Return actor weights and fixed physical support, never inferred identities."""
    x = np.asarray(nodes, dtype=np.float32)
    if x.ndim != 4 or x.shape[1:] != (2, 16, 21) or not np.isfinite(x).all():
        raise ValueError('Invalid player nodes')
    valid = x[..., 20] > .5
    if model is None:
        weights = np.exp(-10 * x[..., 8]) * valid
    else:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(x), 512):
                chunks.append(actor_logits(model, torch.from_numpy(x[start:start+512])).softmax(-1).numpy())
        weights = np.concatenate(chunks).reshape(-1, 2, 16) * valid if len(x) else np.empty((0, 2, 16))
    weights = weights / np.maximum(weights.sum((1, 2), keepdims=True), 1e-8)
    distance = np.expm1(np.maximum(0, 5 * x[..., 8]))
    proximity = np.minimum(1, 2 * np.exp(-distance / 2))
    relative_speed = np.linalg.norm(x[..., 4:6] * 20, axis=-1)
    coupling = np.exp(-relative_speed / 5) * (x[..., 9] > .5)
    compatibility = np.maximum.reduce([coupling, x[..., 14], x[..., 16], x[..., 18], x[..., 19]])
    physical = proximity * (.2 + .8 * np.clip(compatibility, 0, 1))
    support = np.sqrt(np.maximum(0, weights * physical)) * valid * (x[..., 10] > .5)
    return weights, support.max(2)


def decode_ledger(team_probability, support, frozen_labels, preserve_unsupported=False):
    """Four latent states HC/Hflight/AC/Aflight; labels 0=H,1=A,2=D.

    No minimum duration. Unsupported opponent flight costs 24; supported
    receipts progressively lower handover cost. Dead frames are immutable.
    """
    q = np.asarray(team_probability, dtype=float)
    r = np.asarray(support, dtype=float)
    frozen = np.asarray(frozen_labels)
    if q.shape != (len(frozen), 2) or r.shape != q.shape:
        raise ValueError('Ledger input shape mismatch')
    if not np.isfinite(q).all() or not np.isfinite(r).all() or (q < 0).any() or ((r < 0) | (r > 1)).any() or not np.isin(frozen, [0, 1, 2]).all():
        raise ValueError('Invalid ledger inputs')
    q = q / np.maximum(q.sum(1, keepdims=True), 1e-8)
    flight = 1 - r.max(1)
    emissions = np.maximum(np.column_stack([q[:, 0]*r[:, 0], q[:, 0]*flight, q[:, 1]*r[:, 1], q[:, 1]*flight]), 1e-8)
    labels = frozen.copy(); latent = np.full(len(frozen), -1, np.int8)
    live = frozen != 2
    if preserve_unsupported:
        # No association evidence grants no authority to revise the prior.
        live = live & (r.max(1) > 0)
    edges = np.flatnonzero(np.diff(np.r_[False, live, False]))
    teams = np.array([0, 0, 1, 1])
    for start, end in edges.reshape(-1, 2):
        score = np.log(emissions[start]); back = np.zeros((end-start, 4), np.int8)
        for t in range(start+1, end):
            cost = np.zeros((4, 4))
            for source in range(4):
                for dest in range(4):
                    if source == dest:
                        continue
                    if teams[source] == teams[dest]:
                        cost[source, dest] = .25
                    else:
                        cost[source, dest] = 24 if dest % 2 else 2 + 22*(1-r[t, teams[dest]])
            candidates = score[:, None] - cost
            back[t-start] = candidates.argmax(0)
            score = candidates.max(0) + np.log(emissions[t])
        state = int(score.argmax())
        for t in range(end-1, start-1, -1):
            latent[t] = state; labels[t] = teams[state]
            if t > start:
                state = int(back[t-start, state])
    return labels, latent
