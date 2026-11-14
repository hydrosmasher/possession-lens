import unittest
import numpy as np
from controlswitch.spell_training import spell_weights

class SpellTrainingWeightsTests(unittest.TestCase):
    def test_short_possession_gains_weight_and_dead_unknown_are_excluded(self):
        y=np.r_[np.zeros(100,int),np.ones(10,int),np.full(20,2),np.full(10,3)]
        w=spell_weights(y)
        self.assertGreater(w[100],w[0]);self.assertAlmostEqual(w[y<2].mean(),1.)
        self.assertTrue((w[y>=2]==0).all());self.assertTrue(np.isfinite(w).all())
    def test_weight_is_team_invariant_and_ignores_absolute_time(self):
        y=np.r_[np.zeros(100,int),np.ones(10,int),np.full(20,2)]
        swap=np.where(y<2,1-y,y)
        np.testing.assert_array_equal(spell_weights(y),spell_weights(swap))
        np.testing.assert_array_equal(spell_weights(np.r_[np.full(30,3),y])[30:],spell_weights(y))
    def test_no_live_frames_and_invalid_input(self):
        np.testing.assert_array_equal(spell_weights(np.array([2,3,2])),np.zeros(3))
        for x in (np.array([]),np.array([-1]),np.array([[0]])):
            with self.assertRaises(ValueError):spell_weights(x)
if __name__=='__main__':unittest.main()
