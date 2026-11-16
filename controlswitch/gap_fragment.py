"""081: avoid adding a short fragment next to unsupported missing-ball control."""
import numpy as np


def filter_gap_fragments(frozen, current, events, seen, receipt_support):
    """Keep original missing-frame labels; veto only a novel isolated pulse.

    A nearby existing same-team spell and a <=3s majority-missing intervening
    bridge without ANY opposing receipt support suggest fragmentation rather
    than a new receipt. This is an uncertain continuity proxy, not touch truth.
    """
    frozen, current = np.asarray(frozen), np.asarray(current)
    seen, support = np.asarray(seen, bool), np.asarray(receipt_support, bool)
    n = len(frozen)
    if frozen.shape != (n,) or current.shape != (n,) or seen.shape != (n,) or support.shape != (n, 2) or not np.isin(frozen, [0, 1, 2]).all() or not np.isin(current, [0, 1, 2]).all():
        raise ValueError('Invalid fragment evidence')
    result = current.copy(); reviews = []; end = 0
    for e in events:
        a, b, k = e['start'], e['end'], e['team']; other = 1-k
        if not 0 < a < b < n or b-a > 75 or a < end or k not in (0, 1) or not (frozen[a-1:b+1] == other).all() or not (current[a:b] == k).all():
            raise ValueError('Fragment guard requires a novel bounded proposal')
        end = b
        left, right = a, b
        while left > 0 and frozen[left-1] == other: left -= 1
        while right < n and frozen[right] == other: right += 1
        bridges = []
        for side, u, v, adjacent in (('left', left, a, left-1), ('right', b, right, right)):
            if not 0 <= adjacent < n or frozen[adjacent] != k or not 0 < v-u <= 75: continue
            missing = float((~seen[u:v]).mean())
            opposition = bool(support[u:v, other].any())
            bridges.append({'side': side, 'start': int(u), 'end': int(v), 'missing_fraction': missing,
                            'opposing_receipt_support': opposition, 'fragment_risk': bool(missing >= .5 and not opposition)})
        veto = any(v['fragment_risk'] for v in bridges)
        if veto: result[a:b] = frozen[a:b]
        reviews.append({**e, 'bridges': bridges, 'veto': veto})
    return result, reviews
