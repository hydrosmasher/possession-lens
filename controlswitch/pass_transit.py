"""Offline, observation-only pass-transit veto; never a verified-touch detector."""
import numpy as np


def veto_pass_transits(labels, sequence, mask=None):
    """Remove only isolated <=3s H/A pulses with conservative pass evidence.

    Fixed 25Hz geometry, no fitted Barcelona/team prior. Two different players
    of the surrounding team must bracket a straight, steady fast flight. The
    closest proposed opponent must not move with that flight. Unknown evidence
    abstains; no gap interpolation, new labels, or changes to dead-ball frames.
    """
    y = np.asarray(labels)
    ball = np.asarray(sequence['ball'], dtype=float)
    xy = np.asarray(sequence['xy'], dtype=float)
    teams = np.asarray(sequence['teams'])
    n = len(y)
    if (y.shape != (n,) or ball.shape != (n, 2) or xy.ndim != 3
            or xy.shape[0] != n or xy.shape[2] != 2
            or teams.shape != (xy.shape[1],) or not np.isin(y, [0, 1, 2]).all()
            or not np.isin(teams, [0, 1]).all()):
        raise ValueError('Invalid transit observations')
    seen = np.isfinite(ball).all(1)
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != (n,):
            raise ValueError('Invalid ball mask')
        seen &= ~mask
    out = y.copy()
    records = []
    if not n:
        return out, records
    edges = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, n]
    for a, b in zip(edges[:-1], edges[1:]):
        team = int(y[a])
        if team == 2 or a == 0 or b == n or b-a > 75:
            continue
        other = 1-team
        if y[a-1] != other or y[b] != other:
            continue
        lo, hi = max(0, a-25), min(n, b+25)
        record = {'start': int(a), 'end': int(b), 'team': team, 'veto': False}
        records.append(record)
        if not seen[lo:hi].all():
            continue
        slots = np.flatnonzero(teams == other)
        opponents = np.flatnonzero(teams == team)
        if not len(slots) or not len(opponents):
            continue
        d = np.linalg.norm(xy[lo:hi, slots]-ball[lo:hi, None], axis=2)
        d = np.where(np.isfinite(d), d, np.inf)
        # Endpoint ownership evidence must agree with the surrounding labels.
        left = np.where((y[lo:a] == other)[:, None], d[:a-lo], np.inf)
        right = np.where((y[b:hi] == other)[:, None], d[b-lo:], np.inf)
        il, jl = np.unravel_index(left.argmin(), left.shape)
        ir, jr = np.unravel_index(right.argmin(), right.shape)
        if left[il, jl] > 2.5 or right[ir, jr] > 2.5 or jl == jr:
            continue
        l, r = lo+il, b+ir
        if r-l < 12 or (y[l:r+1] == 2).any():
            continue
        t = np.arange(r-l+1, dtype=float)/25
        centered = t-t.mean()
        flight = ball[l:r+1]
        velocity = (centered[:, None]*(flight-flight.mean(0))).sum(0)/(centered**2).sum()
        speed = float(np.linalg.norm(velocity))
        residual = float(np.linalg.norm(flight-(flight.mean(0)+centered[:, None]*velocity), axis=1).max())
        local_v = (flight[6:]-flight[:-6])/.24
        velocity_error = float(np.linalg.norm(local_v-velocity, axis=1).max())
        if speed < 4 or residual > .75 or velocity_error > max(2., .25*speed):
            continue
        # Follow the SAME nearby opponent, rather than swapping nearest slots.
        od = np.linalg.norm(xy[a:b, opponents]-ball[a:b, None], axis=2)
        od = np.where(np.isfinite(od), od, np.inf)
        it, jt = np.unravel_index(od.argmin(), od.shape)
        k, slot = a+it, opponents[jt]
        if od[it, jt] > 2.5 or k < 6 or k+6 >= n:
            continue
        player = xy[k-6:k+7, slot]
        if not np.isfinite(player).all():
            continue
        player_v = (player[-1]-player[0])/.48
        relative_speed = float(np.linalg.norm(velocity-player_v))
        if relative_speed < 4:
            continue
        record.update(veto=True, pass_start=int(l), pass_end=int(r),
                      speed=speed, max_line_residual=residual,
                      max_velocity_error=velocity_error, relative_speed=relative_speed)
        out[a:b] = other
    return out, records
