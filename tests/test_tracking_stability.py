import unittest
import numpy as np
from controlswitch.tracking_stability import deduplicate, median_three, smooth_visible, filter_unstable


class TrackingStabilityTests(unittest.TestCase):
    def sequence(self):
        n = 400; t = np.arange(n)/25
        p = np.column_stack([t*2, np.sin(t)])
        rng = np.random.default_rng(179)
        xy = np.stack([p, p+[20, 0], p+[5, 5], p+rng.normal(0, .1, p.shape)], axis=1)
        return {'xy': xy, 'teams': np.array([0, 0, 1, 0]), 'speed': np.ones((n, 4))*2, 'ball': p.copy(), 'ball_speed': np.ones(n)*2}

    def test_persistent_duplicate_removed_without_team_merge(self):
        s = self.sequence(); result, records = deduplicate({**s, 'y': np.zeros(400)})
        self.assertEqual(result['xy'].shape, (400, 3, 2)); self.assertNotIn('y', result)
        self.assertEqual(records[0]['removed_slot'], 3); self.assertEqual(records[0]['kept_slot'], 0)
        swapped, _ = deduplicate({**s, 'teams': 1-s['teams']})
        np.testing.assert_array_equal(swapped['xy'], result['xy'])
        np.testing.assert_array_equal(swapped['teams'], 1-result['teams'])

    def test_brief_encounter_or_missing_overlap_not_deduplicated(self):
        s = self.sequence(); s['xy'][:300, 3] += 10
        result, records = deduplicate(s)
        self.assertFalse(records); self.assertEqual(result['xy'].shape, s['xy'].shape)
        s = self.sequence(); s['xy'][:200, 3] = np.nan
        _, records = deduplicate(s); self.assertFalse(records)

    def test_median_never_fills_gaps_or_reads_hidden_ball(self):
        a = np.array([0., 50., 1., np.nan, 2., 99., 3.])
        z = median_three(a)
        self.assertEqual(z[1], 1.); self.assertTrue(np.isnan(z[3])); self.assertEqual(z[2], a[2])
        s = self.sequence(); mask = np.zeros(400, bool); mask[200] = True
        a = smooth_visible(s, mask); s['ball'][200] = [1e6, 1e6]; b = smooth_visible(s, mask)
        np.testing.assert_array_equal(a['ball'], b['ball'])

    def test_only_new_unsupported_recovery_reverts(self):
        frozen = np.zeros(100, int); frozen[80:] = 2
        current = frozen.copy(); current[30:40] = 1
        events = [{'start': 30, 'end': 40, 'team': 1}]
        support = np.zeros((100, 2), bool); support[35, 1] = True
        raw = np.tile([.2, .8], (100, 1)); cleaned = raw[:, ::-1].copy()
        out, r = filter_unstable(frozen, current, events, support, raw, cleaned, True)
        np.testing.assert_array_equal(out, frozen); self.assertTrue(r[0]['veto'])
        for duplicates, scores in [(False, cleaned), (True, raw), (True, np.full_like(raw, .5))]:
            out, _ = filter_unstable(frozen, current, events, support, raw, scores, duplicates)
            np.testing.assert_array_equal(out, current)
