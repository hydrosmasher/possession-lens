import unittest
import numpy as np
from controlswitch.short_preservation import preservation_anchors
from controlswitch.continuity_refinement import anchored_decode


class ShortPreservationTests(unittest.TestCase):
    def test_short_spell_and_both_flanks_survive_adverse_neural_evidence(self):
        frozen = np.ones(500, int); frozen[200:300] = 0; frozen[450:] = 2
        anchors, protected = preservation_anchors(frozen, frozen, np.ones(500, bool))
        q = np.tile([.01,.99], (500,1))
        p = anchored_decode(q, np.ones(500), frozen, anchors)
        np.testing.assert_array_equal(p[199:301], frozen[199:301])
        self.assertIn({'start':200,'end':300,'team':0}, protected)
        np.testing.assert_array_equal(p == 2, frozen == 2)

    def test_missing_and_one_frame_recovery_preserved_without_labels(self):
        frozen = np.zeros(500, int); candidate = frozen.copy(); candidate[300] = 1
        seen = np.ones(500, bool); seen[100:200] = False
        anchors, _ = preservation_anchors(frozen, candidate, seen)
        self.assertTrue((anchors[100:200] == 0).all())
        np.testing.assert_array_equal(anchors[299:302], [0,1,0])
        empty, records = preservation_anchors(frozen[:0], candidate[:0], seen[:0])
        self.assertFalse(len(empty)); self.assertFalse(records)
