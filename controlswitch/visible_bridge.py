"""082: reconnect recovered fragments over visible, model-supported bridges."""
import numpy as np


def join_visible_bridges(frozen, current, events, seen, prior, local, boundary):
    """Fixed equal-expert log likelihood plus existing adaptive switch costs.

    A join must attach to existing same-team control over at most75 observed
    frames. It removes two boundaries; select at most one side per proposal.
    Existing short spells can be harmed, so exact spell-retention checks remain
    mandatory externally. This objective never reads reference possessions.
    """
    frozen, current = np.asarray(frozen), np.asarray(current)
    seen, prior, local, boundary = np.asarray(seen, bool), np.asarray(prior, float), np.asarray(local, float), np.asarray(boundary, float)
    n = len(frozen)
    if (frozen.shape != (n,) or current.shape != (n,) or seen.shape != (n,)
            or prior.shape != (n, 2) or local.shape != (n, 2) or boundary.shape != (n,)
            or not np.isin(frozen, [0, 1, 2]).all() or not np.isin(current, [0, 1, 2]).all()
            or not all(np.isfinite(a).all() for a in (prior, local, boundary))
            or (prior < 0).any() or (local < 0).any() or (prior.sum(1) <= 0).any() or (local.sum(1) <= 0).any()
            or ((boundary < 0) | (boundary > 1)).any()):
        raise ValueError('Invalid bridge evidence')
    q = .5*prior/prior.sum(1, keepdims=True)+.5*local/local.sum(1, keepdims=True)
    logq = np.log(np.maximum(q, 1e-8))
    costs = 12*np.clip(2-2*boundary, .25, 2)
    out = current.copy(); reviews = []; end = 0
    for e in events:
        a, b, k = e['start'], e['end'], e['team']; other = 1-k
        if not 0 < a < b < n or b-a > 75 or a < end or k not in (0, 1) or not (frozen[a-1:b+1] == other).all() or not (current[a:b] == k).all():
            raise ValueError('Bridge joining requires a novel bounded recovery')
        end = b
        left, right = a, b
        while left > 0 and frozen[left-1] == other: left -= 1
        while right < n and frozen[right] == other: right += 1
        choices = []
        for side, u, v, adjacent in (('left', left, a, left-1), ('right', b, right, right)):
            if not 0 <= adjacent < n or frozen[adjacent] != k or not 0 < v-u <= 75: continue
            if not seen[u:v].all() or not (out[u:v] == other).all(): continue
            likelihood_gain = float((logq[u:v, k]-logq[u:v, other]).sum())
            boundary_gain = float(costs[u]+costs[v])
            choices.append({'side': side, 'start': int(u), 'end': int(v), 'likelihood_gain': likelihood_gain,
                            'boundary_gain': boundary_gain, 'objective_gain': likelihood_gain+boundary_gain})
        positive = [c for c in choices if c['objective_gain'] > 0]
        chosen = max(positive, key=lambda c: c['objective_gain']) if positive else None
        if chosen: out[chosen['start']:chosen['end']] = k
        reviews.append({**e, 'choices': choices, 'join': chosen})
    return out, reviews
