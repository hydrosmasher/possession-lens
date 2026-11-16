import unittest
import numpy as np
from controlswitch.noise_agreement import filter_prior_agreement


class NoiseAgreementTests(unittest.TestCase):
    def inputs(self):
        frozen = np.zeros(120, int); frozen[100:] = 2
        current = frozen.copy(); current[30:40] = 1
        events = [{'start': 30, 'end': 40, 'team': 1}]
        support = np.zeros((120, 2), bool); support[35, 1] = True
        prior = np.tile([.8, .2], (120, 1))
        return frozen, current, events, support, prior

    def test_only_affected_unsupported_novel_recovery_reverts(self):
        a = self.inputs(); p, r = filter_prior_agreement(*a, True)
        np.testing.assert_array_equal(p, a[0]); self.assertTrue(r[0]['veto'])
        p, _ = filter_prior_agreement(*a, False); np.testing.assert_array_equal(p, a[1])
        a[3][:] = False
        p, _ = filter_prior_agreement(*a, True); np.testing.assert_array_equal(p, a[1])

    def test_tolerance_handles_timing_offset_inside_candidate_only(self):
        a = self.inputs(); a[4][33] = [.2, .8]
        exact, _ = filter_prior_agreement(*a, True, 0)
        tolerant, _ = filter_prior_agreement(*a, True, 3)
        np.testing.assert_array_equal(exact, a[0]); np.testing.assert_array_equal(tolerant, a[1])
        a[4][:] = [.8, .2]; a[4][40] = [.2, .8]; a[3][:] = False; a[3][39, 1] = True
        p, _ = filter_prior_agreement(*a, True, 3); np.testing.assert_array_equal(p, a[0])

    def test_single_frame_receipt_and_team_swap(self):
        a = list(self.inputs()); a[1][30:40] = 0; a[1][35] = 1
        a[2] = [{'start': 35, 'end': 36, 'team': 1}]; a[4][35] = [.2, .8]
        p, _ = filter_prior_agreement(*a, True)
        np.testing.assert_array_equal(p, a[1])
        old, current = a[0].copy(), a[1].copy()
        old[old < 2] = 1-old[old < 2]; current[current < 2] = 1-current[current < 2]
        q, _ = filter_prior_agreement(old, current, [{'start':35,'end':36,'team':0}], a[3][:, ::-1], a[4][:, ::-1], True)
        expected = p.copy(); expected[p < 2] = 1-p[p < 2]
        np.testing.assert_array_equal(q, expected)
