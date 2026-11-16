import unittest
import numpy as np
from controlswitch.receipt_context import context_features, gate_recoveries, short_retention_threshold, FEATURE_NAMES
from controlswitch.receipt_weighted_experiment import fitting_weights


class ReceiptContextTests(unittest.TestCase):
    def observations(self):
        t = np.arange(160)/25
        ball = np.column_stack([t*6, np.sin(t)])
        xy = np.stack([ball+[.2, 0], ball+[0, 2], ball+[4, 0], ball+[5, 2]], axis=1)
        return {'xy': xy, 'ball': ball, 'teams': np.array([0, 0, 1, 1])}

    def test_team_swap_and_reference_independence(self):
        s = self.observations(); x, v = context_features(s)
        self.assertEqual(x.shape, (160, 2, len(FEATURE_NAMES)))
        self.assertTrue(v[24:-24].all()); self.assertFalse(v[:24].any())
        z, w = context_features({**s, 'teams': 1-s['teams'], 'y': np.full(160, 2)})
        np.testing.assert_array_equal(z, x[:, ::-1]); np.testing.assert_array_equal(w, v[:, ::-1])

    def test_missing_ball_and_same_slot_gap_abstain(self):
        s = self.observations(); mask = np.zeros(160, bool); mask[80] = True
        _, v = context_features(s, mask)
        self.assertFalse(v[56:105].any())
        s['xy'][80, 0] = np.nan
        x, v = context_features(s)
        self.assertFalse(v[79, 0]); self.assertTrue(np.isfinite(x).all())

    def test_only_new_proposal_changes_and_unknown_support_retained(self):
        current = np.zeros(120, int); current[100:] = 2
        events = [{'start': 30, 'end': 40, 'team': 1}]
        support = np.zeros((120, 2), bool); support[33:35, 1] = True
        valid = np.ones_like(support); score = np.zeros((120, 2))
        p, r = gate_recoveries(current, events, support, valid, score, .5)
        np.testing.assert_array_equal(p, current); self.assertFalse(r[0]['kept'])
        score[34, 1] = .5
        p, _ = gate_recoveries(current, events, support, valid, score, .5)
        self.assertTrue((p[30:40] == 1).all())
        score[:] = 0; valid[33, 1] = False
        q, r = gate_recoveries(current, events, support, valid, score, .5)
        np.testing.assert_array_equal(q, p); self.assertFalse(r[0]['context_complete'])
        np.testing.assert_array_equal(q[:30], current[:30]); np.testing.assert_array_equal(q[40:], current[40:])

    def test_calibration_and_empty_input(self):
        self.assertEqual(short_retention_threshold([]), 0.)
        self.assertEqual(short_retention_threshold([.7, .8, .2]), .2)
        with self.assertRaises(ValueError): short_retention_threshold([np.nan])
        s = self.observations(); x, v = context_features({**s, 'xy': s['xy'][:0], 'ball': s['ball'][:0]})
        self.assertEqual(len(x), 0); self.assertEqual(len(v), 0)

    def test_short_positive_and_close_decoy_training_weights(self):
        y = np.zeros(120, int); y[10:20] = 1
        frames = np.array([15, 15, 90, 90]); teams = np.array([1, 0, 0, 1])
        x = np.zeros((4, len(FEATURE_NAMES)))
        x[:, FEATURE_NAMES.index('opponent_distance')] = [9, 2, 3, 9]
        np.testing.assert_array_equal(fitting_weights(y, frames, teams, x), [4, 2, 1, 1])
