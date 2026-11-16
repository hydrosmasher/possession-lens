import numpy as np
import unittest
from controlswitch.pass_transit import veto_pass_transits


def example():
    n = 120
    ball = np.column_stack([np.arange(n)*.4, np.zeros(n)])
    xy = np.zeros((n, 3, 2))
    xy[:, 0] = [12., 0.]
    xy[:, 1] = [32., 0.]
    xy[:, 2] = [22., .3]
    y = np.zeros(n, int)
    y[48:63] = 1
    return y, {'ball': ball, 'xy': xy, 'teams': np.array([0, 0, 1])}


def check_straight_flight():
    y, s = example()
    p, events = veto_pass_transits(y, s)
    assert not p.any() and sum(e['veto'] for e in events) == 1
    swapped = {**s, 'teams': 1-s['teams']}
    q, _ = veto_pass_transits(1-y, swapped)
    np.testing.assert_array_equal(q, 1-p)


def check_abstention(kind):
    y, s = example()
    mask = None
    if kind == 'missing': s['ball'][51] = np.nan
    if kind == 'masked':
        mask = np.zeros(len(y), bool); mask[51] = True
    if kind == 'turn': s['ball'][55:, 1] += np.arange(len(y)-55)*.35
    if kind == 'following': s['xy'][:, 2] = s['ball'] + [.1, .3]
    if kind == 'dead': y[63] = 2
    if kind == 'same_endpoint': s['xy'][:, 1] = s['xy'][:, 0]
    if kind == 'slow':
        s['ball'] *= .2; s['xy'] *= .2
    p, _ = veto_pass_transits(y, s, mask)
    np.testing.assert_array_equal(p, y)


def check_empty_and_bad_mask():
    y, s = example()
    with np.testing.assert_raises(ValueError): veto_pass_transits(y, s, np.zeros(2, bool))
    p, events = veto_pass_transits(y[:0], {**s, 'ball': s['ball'][:0], 'xy': s['xy'][:0]})
    assert len(p) == 0 and not events


class PassTransitTests(unittest.TestCase):
    def test_straight_flight_and_team_swap(self):
        check_straight_flight()

    def test_uncertain_evidence_abstains(self):
        for kind in ('missing', 'masked', 'turn', 'following', 'dead', 'same_endpoint', 'slow'):
            with self.subTest(kind=kind):
                check_abstention(kind)

    def test_empty_and_bad_mask(self):
        check_empty_and_bad_mask()
