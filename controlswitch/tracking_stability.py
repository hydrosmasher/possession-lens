"""079: conservative duplicate-track cleanup and recovery stability checks."""
import numpy as np


def deduplicate(sequence):
    """Identify persistent same-team near-identical tracks, not player names.

    At least250co-visible frames, >=95% visibility-mask overlap, and95th
    percentile separation<=0.5m. Keep the most-visible/smoothest trajectory.
    No coordinates or reference labels are used to infer jersey identities.
    """
    xy = np.asarray(sequence['xy'], float)
    teams = np.asarray(sequence['teams'])
    n = len(xy)
    if xy.ndim != 3 or xy.shape[2] != 2 or teams.shape != (xy.shape[1],) or not np.isin(teams, [0, 1]).all():
        raise ValueError('Invalid player observations')
    valid = np.isfinite(xy).all(2)
    roughness = []
    for j in range(xy.shape[1]):
        acceleration = np.linalg.norm(xy[2:, j]-2*xy[1:-1, j]+xy[:-2, j], axis=1)
        finite = acceleration[np.isfinite(acceleration)]
        roughness.append(float(np.median(finite)) if len(finite) else np.inf)
    order = sorted(range(xy.shape[1]), key=lambda j: (-int(valid[:, j].sum()), roughness[j], j))
    kept, removed = [], []
    for j in order:
        duplicate = None
        for k in kept:
            if teams[k] != teams[j]: continue
            overlap = valid[:, k] & valid[:, j]
            union = valid[:, k] | valid[:, j]
            if overlap.sum() < 250 or overlap.sum() < .95*union.sum(): continue
            separation = np.linalg.norm(xy[overlap, k]-xy[overlap, j], axis=1)
            distance95 = float(np.quantile(separation, .95))
            if distance95 <= .5:
                duplicate = {'removed_slot': int(j), 'kept_slot': int(k), 'team': int(teams[j]),
                             'co_visible_frames': int(overlap.sum()), 'distance95_m': distance95}
                break
        if duplicate is None: kept.append(j)
        else: removed.append(duplicate)
    kept = np.array(sorted(kept), int)
    result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sequence.items() if k != 'y'}
    result['xy'] = xy[:, kept].copy()
    result['teams'] = teams[kept].copy()
    if 'speed' in sequence:
        if np.asarray(sequence['speed']).shape != (n, len(teams)): raise ValueError('Invalid player speeds')
        result['speed'] = np.asarray(sequence['speed'])[:, kept].copy()
    return result, removed


def median_three(values):
    """Three-frame median only where all neighbours exist; never fill a gap."""
    a = np.asarray(values, float)
    out = a.copy()
    if len(a) < 3: return out
    windows = np.stack([a[:-2], a[1:-1], a[2:]], axis=1)
    complete = np.isfinite(windows).all(1)
    out[1:-1] = np.where(complete, np.median(windows, axis=1), a[1:-1])
    np.testing.assert_array_equal(np.isfinite(out), np.isfinite(a))
    return out


def smooth_visible(sequence, mask=None):
    result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sequence.items() if k != 'y'}
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != (len(result['ball']),): raise ValueError('Invalid ball mask')
        result['ball'][mask] = np.nan
        if 'ball_speed' in result: result['ball_speed'][mask] = np.nan
    for key in ('xy', 'speed', 'ball', 'ball_speed'):
        if key in result: result[key] = median_three(result[key])
    return result


def filter_unstable(frozen, current, events, support, raw, cleaned, has_duplicates):
    """Veto only a new recovery whose local evidence reverses after cleanup.

    A candidate has to have raw-majority support on an old receipt-support
    frame, and cleaned evidence must favour the opponent on ALL such frames.
    Ties, absent support, and no detected duplicates all retain the candidate.
    """
    frozen, current = np.asarray(frozen), np.asarray(current)
    support, raw, cleaned = np.asarray(support, bool), np.asarray(raw, float), np.asarray(cleaned, float)
    n = len(frozen)
    if (frozen.shape != (n,) or current.shape != (n,) or support.shape != (n, 2)
            or raw.shape != support.shape or cleaned.shape != support.shape
            or not np.isin(frozen, [0, 1, 2]).all() or not np.isin(current, [0, 1, 2]).all()
            or not np.isfinite(raw).all() or not np.isfinite(cleaned).all()
            or (raw < 0).any() or (cleaned < 0).any()
            or not np.allclose(raw.sum(1), 1) or not np.allclose(cleaned.sum(1), 1)):
        raise ValueError('Invalid stability evidence')
    out = current.copy(); reviews = []; end = 0
    for e in events:
        a, b, k = e['start'], e['end'], e['team']
        if not 0 < a < b < n or b-a > 75 or a < end or k not in (0, 1) or not (frozen[a-1:b+1] == 1-k).all() or not (current[a:b] == k).all():
            raise ValueError('Stability check requires a novel bounded proposal')
        end = b
        good = support[a:b, k]
        raw_peak = float(raw[a:b, k][good].max()) if good.any() else None
        cleaned_peak = float(cleaned[a:b, k][good].max()) if good.any() else None
        veto = bool(has_duplicates and good.any() and raw_peak > .5 and cleaned_peak < .5)
        if veto: out[a:b] = frozen[a:b]
        reviews.append({**e, 'raw_peak': raw_peak, 'cleaned_peak': cleaned_peak, 'veto': veto})
    return out, reviews
