"""Observation-only, offline receipt compatibility (not verified touch detection)."""
import numpy as np

OFFSETS = (-12, -6, -3, 0, 3, 6, 12)
FEATURE_NAMES = ('distance', 'distance_before12', 'distance_before6', 'distance_before3',
                 'distance_after3', 'distance_after6', 'distance_after12',
                 'relative_speed_before', 'relative_speed_after', 'ball_speed_before',
                 'ball_speed_after', 'speed_drop', 'trajectory_turn', 'future_close_fraction')


def receipt_features(sequence, mask=None):
    """Track the same nearest current player into +/-0.48s, with no gap fill.

    Only ball/xy/teams enter features. Reject unavailable/implausible trajectory
    windows rather than treating missing velocities as a controlled receipt.
    """
    xy = np.asarray(sequence['xy'], dtype=float)
    ball = np.array(sequence['ball'], dtype=float, copy=True)
    teams = np.asarray(sequence['teams'])
    n = len(ball)
    if xy.ndim != 3 or xy.shape[0] != n or xy.shape[2] != 2 or teams.shape != (xy.shape[1],):
        raise ValueError('Invalid receipt observations')
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != (n,): raise ValueError('Invalid ball mask')
        ball[mask] = np.nan
    output = np.zeros((n, 2, len(FEATURE_NAMES)), np.float32)
    available = np.zeros((n, 2), bool)
    if not n: return output, available
    t = np.arange(n)
    offsets = np.asarray(OFFSETS)
    indices = np.clip(t[:, None] + offsets, 0, n-1)
    b = ball[indices]
    in_bounds = (t >= 12) & (t < n-12)
    # Every intervening ball observation must exist; no bridging tracking gaps.
    seen = np.isfinite(ball).all(1)
    c = np.r_[0, np.cumsum(~seen)]
    complete_ball = c[np.minimum(n, t+13)] == c[np.maximum(0, t-12)]
    for team in (0, 1):
        slots = np.flatnonzero(teams == team)
        if not len(slots): raise ValueError('Both teams require tracking slots')
        distances = np.linalg.norm(xy[:, slots]-ball[:, None], axis=2)
        order = np.where(np.isfinite(distances), distances, np.inf).argmin(1)
        slot = slots[order]
        p = xy[indices, slot[:, None]]
        d = np.linalg.norm(p-b, axis=2)
        bv0 = (b[:, 3]-b[:, 1]) / .24
        bv1 = (b[:, 5]-b[:, 3]) / .24
        pv0 = (p[:, 3]-p[:, 1]) / .24
        pv1 = (p[:, 5]-p[:, 3]) / .24
        bs0 = np.linalg.norm(bv0, axis=1); bs1 = np.linalg.norm(bv1, axis=1)
        ps0 = np.linalg.norm(pv0, axis=1); ps1 = np.linalg.norm(pv1, axis=1)
        good = in_bounds & complete_ball & np.isfinite(p).all((1, 2)) & (bs0 < 60) & (bs1 < 60) & (ps0 < 15) & (ps1 < 15)
        turn = np.clip(1-np.sum(bv0*bv1, axis=1)/np.maximum(bs0*bs1, 1e-6), 0, 2)/2
        turn[(bs0 < .5) | (bs1 < .5)] = 0
        cols = [d[:, 3], *[d[:, j] for j in (0, 1, 2, 4, 5, 6)],
                np.linalg.norm(bv0-pv0, axis=1), np.linalg.norm(bv1-pv1, axis=1),
                bs0, bs1, bs0-bs1, turn, (d[:, 4:] <= 2.5).mean(1)]
        x = np.column_stack(cols)
        output[:, team] = np.where(good[:, None], np.clip(np.nan_to_num(x), -100, 100), 0)
        available[:, team] = good
    return output, available


def suppress_unsupported_pulses(labels, scores, available, threshold, max_frames=25):
    """Only revisit isolated <=1s H/A pulses; never invent a new switch.

    Require valid trajectory evidence throughout the pulse and two flanks.
    Unsupported frames and D labels are immutable. Short real receipts with
    any above-threshold compatibility remain unchanged. This is a research
    filter, not a minimum-duration rule.
    """
    y = np.asarray(labels); p = np.asarray(scores); valid = np.asarray(available, bool)
    if p.shape != (len(y), 2) or valid.shape != p.shape or not np.isin(y, [0, 1, 2]).all() or not np.isfinite(p).all() or not 0 <= threshold <= 1:
        raise ValueError('Invalid pulse evidence')
    result = y.copy(); changes = []
    edges = np.r_[0, np.flatnonzero(y[1:] != y[:-1])+1, len(y)]
    for a, b in zip(edges[:-1], edges[1:]):
        owner = int(y[a])
        if owner == 2 or a == 0 or b == len(y) or b-a > max_frames: continue
        other = 1-owner
        if y[a-1] != other or y[b] != other: continue
        if not valid[a-1:b+1].all(): continue
        if p[a:b, owner].max() >= threshold: continue
        if min(p[a-1, other], p[b, other]) < threshold: continue
        result[a:b] = other
        changes.append({'start':int(a), 'end':int(b), 'from':owner, 'to':other})
    return result, changes
