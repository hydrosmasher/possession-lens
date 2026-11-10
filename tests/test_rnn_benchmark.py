import unittest
import numpy as np
from controlswitch.model import decode
from controlswitch.rnn_benchmark import augmented_sequences,partitions

class DecoderProtocolTests(unittest.TestCase):
 def test_penalty_removes_isolated_flip_but_keeps_sustained_change(self):
    p=np.tile([.98,.01,.01],(80,1));p[10]=[.01,.98,.01];p[40:]=[.01,.98,.01]
    raw=decode(p,0);smoothed=decode(p,6)
    self.assertEqual(int((raw[1:]!=raw[:-1]).sum()),3)
    self.assertEqual(int((smoothed[1:]!=smoothed[:-1]).sum()),1)
    self.assertTrue((smoothed[:40]==0).all());self.assertTrue((smoothed[40:]==1).all())
 def test_augmentation_swaps_only_team_labels_and_preserves_unknown(self):
    s={'xy':np.zeros((10,2,2)),'speed':np.zeros((10,2)),'teams':np.array([0,1]),
       'ball':np.zeros((10,2)),'ball_speed':np.zeros(10),'y':np.array([0,1,2,3,0,1,2,3,0,1])}
    original=s['y'].copy();seq=augmented_sequences([(s,np.array([105.,68.]))])
    np.testing.assert_array_equal(seq[2][1],[1,0,2,3,1,0,2,3,1,0])
    np.testing.assert_array_equal(original,s['y'])
    self.assertTrue(np.isfinite(seq[1][0]).all())
 def test_incomplete_matches_never_become_fresh_test(self):
    with self.assertRaises(ValueError):partitions({'J03WOH':{},'J03WMX':{}},True)
