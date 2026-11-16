"""077: longer same-slot motion and competing-team receipt compatibility."""
import numpy as np
from .receipt_evidence import receipt_features, FEATURE_NAMES as BASE_NAMES
from .proposal_calibration import apply_scores

EXTRA_NAMES = (
    'distance_before24', 'distance_after24', 'past_close_fraction24',
    'future_close_fraction24', 'relative_speed_before48', 'relative_speed_after48',
    'player_speed_before48', 'player_speed_after48', 'ball_speed_before48',
    'ball_speed_after48', 'extended_speed_drop', 'extended_ball_turn',
    'future_velocity_alignment', 'player_motion_turn', 'opponent_distance',
    'distance_advantage', 'opponent_future_distance', 'future_distance_advantage',
    'nearby_teammates', 'future_same_slot_fraction')
FEATURE_NAMES = (*BASE_NAMES, *EXTRA_NAMES)


def context_features(sequence, mask=None):
    """Only coordinates/teams, offline +/-0.96s, without bridging gaps.

    Provider possession labels are never read. Slot continuity is local tracking
    continuity, not a verified player identity or a verified touch annotation.
    """
    base, base_valid = receipt_features(sequence, mask)
    xy = np.asarray(sequence['xy'], dtype=float)
    ball = np.asarray(sequence['ball'], dtype=float)
    teams = np.asarray(sequence['teams'])
    n = len(ball)
    result = np.zeros((n, 2, len(FEATURE_NAMES)), np.float32)
    available = np.zeros((n, 2), bool)
    if not n:
        return result, available
    seen = np.isfinite(ball).all(1)
    if mask is not None: seen &= ~np.asarray(mask, bool)
    t = np.arange(n)
    offsets = np.arange(-24, 25)
    ix = np.clip(t[:, None]+offsets, 0, n-1)
    b = ball[ix]
    cumulative = np.r_[0, np.cumsum(~seen)]
    complete = (cumulative[np.minimum(n, t+25)] == cumulative[np.maximum(0, t-24)]) & (t >= 24) & (t < n-24)
    ball_before = (b[:, 24]-b[:, 12])/.48
    ball_after = (b[:, 36]-b[:, 24])/.48
    bs0 = np.linalg.norm(ball_before, axis=1)
    bs1 = np.linalg.norm(ball_after, axis=1)

    def cosine(a, b):
        return np.clip(np.sum(a*b, axis=1)/np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-6), -1, 1)

    distances = []
    nearest = []
    for team in (0, 1):
        slots = np.flatnonzero(teams == team)
        d = np.linalg.norm(xy[:, slots]-ball[:, None], axis=2)
        d = np.where(np.isfinite(d), d, np.inf)
        distances.append(d)
        nearest.append(slots[d.argmin(1)])
    for team in (0, 1):
        slot = nearest[team]
        p = xy[ix, slot[:, None]]
        d = np.linalg.norm(p-b, axis=2)
        pv0 = (p[:, 24]-p[:, 12])/.48
        pv1 = (p[:, 36]-p[:, 24])/.48
        ps0 = np.linalg.norm(pv0, axis=1)
        ps1 = np.linalg.norm(pv1, axis=1)
        opposite = distances[1-team].min(1)
        opposite_after = opposite[np.clip(t+12, 0, n-1)]
        good = (base_valid[:, team] & complete & np.isfinite(p).all((1, 2))
                & np.isfinite(opposite) & np.isfinite(opposite_after)
                & (bs0 < 60) & (bs1 < 60) & (ps0 < 15) & (ps1 < 15))
        extras = np.column_stack([
            d[:, 0], d[:, -1], (d[:, :24] <= 2.5).mean(1), (d[:, 25:] <= 2.5).mean(1),
            np.linalg.norm(ball_before-pv0, axis=1), np.linalg.norm(ball_after-pv1, axis=1),
            ps0, ps1, bs0, bs1, bs0-bs1, (1-cosine(ball_before, ball_after))/2,
            cosine(ball_after, pv1), (1-cosine(pv0, pv1))/2, opposite,
            opposite-d[:, 24], opposite_after, opposite_after-d[:, 36],
            (distances[team] <= 2.5).sum(1),
            (nearest[team][ix[:, 25:]] == slot[:, None]).mean(1)])
        combined = np.column_stack([base[:, team], extras])
        result[:, team] = np.where(good[:, None], np.clip(np.nan_to_num(combined), -100, 100), 0)
        available[:, team] = good
    return result, available


def gate_recoveries(current, events, old_support, new_valid, scores, threshold):
    """Only remove novel proposals; abstain if any old support is unavailable.

    Any strong contextual support retains the whole short spell. No duration
    floor, smoothing, new D inference, or changes outside the novel proposals.
    """
    n = len(current)
    support = np.asarray(old_support, bool)
    valid = np.asarray(new_valid, bool)
    scores = np.asarray(scores, float)
    if support.shape != (n, 2) or valid.shape != support.shape or scores.shape != support.shape or not np.isfinite(scores).all() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid contextual receipt inputs')
    decisions, review = [], []
    for e in events:
        a, b, k = e['start'], e['end'], e['team']
        if k not in (0, 1) or not 0 < a < b < n:
            raise ValueError('Invalid receipt event')
        good = support[a:b, k]
        complete = bool(good.any() and valid[a:b, k][good].all())
        peak = float(scores[a:b, k][good].max()) if complete else None
        keep = not complete or peak >= threshold
        decisions.append(float(keep))
        review.append({**e, 'context_complete': complete, 'context_peak': peak, 'kept': keep})
    return apply_scores(current, events, np.array(decisions), .5), review


def short_retention_threshold(peaks):
    """All genuine eligible calibration short-spell peaks must survive.

    No positive evidence grants no authority to reject; equality retains.
    """
    p = np.asarray(peaks, float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid short-spell calibration peaks')
    return float(p.min()) if len(p) else 0.
