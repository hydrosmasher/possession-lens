import unittest
import numpy as np
from controlswitch.gap_fragment import filter_gap_fragments


class GapFragmentTests(unittest.TestCase):
    def inputs(self):
        frozen = np.ones(200, int); frozen[:50] = 0; frozen[160:] = 2
        current = frozen.copy(); current[80:95] = 0
        seen = np.ones(200, bool); seen[50:75] = False
        support = np.zeros((200, 2), bool)
        return frozen, current, [{'start':80,'end':95,'team':0}], seen, support

    def test_fragment_veto_keeps_missing_frames_and_old_possessions(self):
        a = self.inputs(); out, events = filter_gap_fragments(*a)
        np.testing.assert_array_equal(out, a[0]); self.assertTrue(events[0]['veto'])
        np.testing.assert_array_equal(out[~a[3]], a[0][~a[3]])

    def test_observed_opponent_receipt_or_visible_bridge_preserves_short_spell(self):
        a = self.inputs(); a[4][76, 1] = True
        out, _ = filter_gap_fragments(*a); np.testing.assert_array_equal(out, a[1])
        a[4][:] = False; a[3][:] = True
        out, _ = filter_gap_fragments(*a); np.testing.assert_array_equal(out, a[1])

    def test_no_crossing_dead_or_long_bridge(self):
        a = self.inputs(); a[0][49] = 2; a[1][49] = 2
        out, _ = filter_gap_fragments(*a); np.testing.assert_array_equal(out, a[1])
        a = self.inputs(); a[0][:50] = 1; a[1][:50] = 1; a[0][0] = a[1][0] = 0
        out, _ = filter_gap_fragments(*a); np.testing.assert_array_equal(out, a[1])
