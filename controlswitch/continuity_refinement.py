"""083: supervised within-spell consistency and constrained offline decoding."""
import numpy as np
import torch


def continuity_loss(logits, labels, weights):
    """Symmetric adjacent KL only within the same KNOWN reference possession.

    Labels are training targets only. No cost across actual changes, unknown
    frames, padding or dead ball; no minimum possession duration is imposed.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or weights.shape != labels.shape:
        raise ValueError('Invalid temporal-consistency batch')
    logp = logits.log_softmax(-1); p = logp.exp()
    kl = .5*((p[:, 1:]*(logp[:, 1:]-logp[:, :-1])).sum(-1)
             +(p[:, :-1]*(logp[:, :-1]-logp[:, 1:])).sum(-1))
    valid = (labels[:, 1:] == labels[:, :-1]) & (labels[:, 1:] >= 0) & (labels[:, 1:] < 2)
    w = torch.minimum(weights[:, 1:], weights[:, :-1])*valid
    return (kl.clamp_min(0)*w).sum()/w.sum().clamp_min(1)


def residual_live(reference, old_gru, new_gru):
    """Exact arithmetic update of a50/50 control/GRU probability fusion."""
    arrays = [np.asarray(a, float) for a in (reference, old_gru, new_gru)]
    if any(a.shape != arrays[0].shape or a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() or (a < 0).any() or (a.sum(1) <= 0).any() for a in arrays):
        raise ValueError('Invalid residual probabilities')
    h, old, new = [a/a.sum(1, keepdims=True) for a in arrays]
    updated = np.clip(h+.5*(new-old), 1e-8, 1)
    return updated/updated.sum(1, keepdims=True)


def anchored_decode(live, event, frozen, anchors):
    """Lock D and selected H/A observations without imposing spell lengths."""
    live, event = np.asarray(live, float), np.asarray(event, float)
    frozen, anchors = np.asarray(frozen), np.asarray(anchors)
    n = len(frozen)
    if (frozen.shape != (n,) or live.shape != (n, 2) or event.shape != (n,) or anchors.shape != (n,)
            or not np.isin(frozen, [0, 1, 2]).all() or not np.isin(anchors, [-1, 0, 1]).all()
            or ((frozen == 2) & (anchors >= 0)).any() or not np.isfinite(live).all() or not np.isfinite(event).all()
            or (live < 0).any() or (live.sum(1) <= 0).any() or ((event < 0) | (event > 1)).any()):
        raise ValueError('Invalid constrained decoder inputs')
    costs = 12*np.clip(2-2*event, .25, 2)
    logp = np.log(np.maximum(live, 1e-8))
    fixed = np.flatnonzero(anchors >= 0)
    logp[fixed, 1-anchors[fixed]] = -np.inf
    out = frozen.copy()
    edges = np.flatnonzero(np.diff(np.r_[False, frozen < 2, False])).reshape(-1, 2)
    for a, b in edges:
        values = logp[a].copy(); back = np.zeros((b-a, 2), np.int8)
        for t in range(a+1, b):
            choices = values[:, None]-np.array([[0, costs[t]], [costs[t], 0]])
            back[t-a] = choices.argmax(0); values = choices.max(0)+logp[t]
        state = int(values.argmax())
        for t in range(b-1, a-1, -1):
            out[t] = state
            if t > a: state = int(back[t-a, state])
    return out
