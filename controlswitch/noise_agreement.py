"""080: require receipt-aligned prior agreement only in duplicate-affected input."""
import numpy as np


def filter_prior_agreement(frozen, current, events, support, prior, affected, tolerance=0):
    frozen, current = np.asarray(frozen), np.asarray(current)
    support, prior = np.asarray(support, bool), np.asarray(prior, float)
    n = len(frozen)
    if (frozen.shape != (n,) or current.shape != (n,) or support.shape != (n, 2)
            or prior.shape != (n, 2) or not np.isfinite(prior).all()
            or (prior < 0).any() or (prior.sum(1) <= 0).any()
            or not np.isin(frozen, [0, 1, 2]).all() or not np.isin(current, [0, 1, 2]).all()
            or tolerance not in (0, 3)):
        raise ValueError('Invalid noise-agreement evidence')
    prior = prior/prior.sum(1, keepdims=True)
    result = current.copy(); reviews = []; end = 0
    for e in events:
        a, b, k = e['start'], e['end'], e['team']
        if not 0 < a < b < n or b-a > 75 or a < end or k not in (0, 1) or not (frozen[a-1:b+1] == 1-k).all() or not (current[a:b] == k).all():
            raise ValueError('Agreement requires novel bounded recovery proposals')
        end = b
        indices = a+np.flatnonzero(support[a:b, k])
        peak = max((float(prior[max(a, t-tolerance):min(b, t+tolerance+1), k].max()) for t in indices), default=None)
        veto = bool(affected and len(indices) and peak <= .5)
        if veto: result[a:b] = frozen[a:b]
        reviews.append({**e, 'receipt_aligned_prior_peak': peak, 'veto': veto})
    return result, reviews
