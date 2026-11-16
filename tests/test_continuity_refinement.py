import unittest
import numpy as np
import torch
from controlswitch.continuity_refinement import continuity_loss, residual_live, anchored_decode
from controlswitch.local_temporal import locked_dead_decode


class ContinuityRefinementTests(unittest.TestCase):
    def test_consistency_penalises_flicker_but_not_true_change_or_unknown(self):
        logits = torch.tensor([[[4., -4., 0.], [-4., 4., 0.]]], requires_grad=True)
        w = torch.ones((1, 2))
        positive = continuity_loss(logits, torch.tensor([[0, 0]]), w)
        self.assertGreater(float(positive.detach()), 0)
        positive.backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        for labels in ([[0, 1]], [[0, -1]], [[2, 2]]):
            self.assertEqual(float(continuity_loss(logits, torch.tensor(labels), w).detach()), 0)

    def test_arithmetic_fusion_and_identity(self):
        base = np.tile([.7, .3], (20, 1)); old = np.tile([.4, .6], (20, 1)); new = np.tile([.8, .2], (20, 1))
        reference = .5*base+.5*old
        np.testing.assert_allclose(residual_live(reference, old, new), .5*base+.5*new)
        np.testing.assert_allclose(residual_live(reference, old, old), reference)

    def test_missing_and_receipt_anchors_and_dead_lock(self):
        frozen = np.zeros(100, int); frozen[70:80] = 2
        live = np.tile([.95, .05], (100, 1)); event = np.ones(100)*.9
        anchors = np.full(100, -1, int); anchors[20:25] = 1; anchors[60] = 0
        p = anchored_decode(live, event, frozen, anchors)
        self.assertTrue((p[20:25] == 1).all()); self.assertEqual(p[60], 0)
        np.testing.assert_array_equal(p == 2, frozen == 2)
        unanchored = np.full(100, -1, int)
        np.testing.assert_array_equal(anchored_decode(live, event, frozen, unanchored), locked_dead_decode(live, event, frozen))

    def test_one_frame_anchor_has_no_duration_floor(self):
        n = 50; frozen = np.zeros(n, int); q = np.tile([.99, .01], (n, 1))
        anchors = np.full(n, -1, int); anchors[25] = 1
        p = anchored_decode(q, np.ones(n), frozen, anchors)
        self.assertEqual(p[25], 1)
