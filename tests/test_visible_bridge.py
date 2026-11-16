import unittest
import numpy as np
from controlswitch.visible_bridge import join_visible_bridges


class VisibleBridgeTests(unittest.TestCase):
    def inputs(self):
        frozen = np.ones(200, int); frozen[:50] = 0; frozen[160:] = 2
        current = frozen.copy(); current[80:95] = 0
        q = np.tile([.8, .2], (200, 1))
        return frozen, current, [{'start':80,'end':95,'team':0}], np.ones(200, bool), q.copy(), q.copy(), np.ones(200)*.9

    def test_supported_bridge_joins_without_new_boundaries(self):
        a = self.inputs(); out, events = join_visible_bridges(*a)
        self.assertTrue((out[:95] == 0).all()); self.assertEqual(events[0]['join']['start'], 50)
        np.testing.assert_array_equal(out[95:], a[1][95:])

    def test_never_changes_missing_or_dead_and_rejects_negative_objective(self):
        a = self.inputs(); a[3][60] = False
        out, _ = join_visible_bridges(*a); np.testing.assert_array_equal(out, a[1])
        a = self.inputs(); a[0][49] = a[1][49] = 2
        out, _ = join_visible_bridges(*a); np.testing.assert_array_equal(out, a[1])
        a = self.inputs(); a[4][:] = [.01, .99]; a[5][:] = [.01, .99]
        out, _ = join_visible_bridges(*a); np.testing.assert_array_equal(out, a[1])

    def test_team_swap_equivariant(self):
        a = self.inputs(); out, _ = join_visible_bridges(*a)
        b = list(a); b[0] = a[0].copy(); b[1] = a[1].copy()
        for x in (b[0], b[1]): x[x < 2] = 1-x[x < 2]
        b[2] = [{'start':80,'end':95,'team':1}]; b[4] = a[4][:, ::-1]; b[5] = a[5][:, ::-1]
        actual, _ = join_visible_bridges(*b); expected = out.copy(); expected[out < 2] = 1-out[out < 2]
        np.testing.assert_array_equal(actual, expected)
