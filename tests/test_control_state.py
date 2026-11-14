import unittest
import numpy as np
from controlswitch.control_state import state_features,swap_state_columns
from controlswitch.control_correction import control_features

class ControlStateFeatureTests(unittest.TestCase):
    def sequence(self):
        rng=np.random.default_rng(7);n=100
        return {'xy':rng.normal(size=(n,4,2)),'speed':np.ones((n,4)),'teams':np.array([0,0,1,1]),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)}
    def test_old_feature_prefix_is_exact_and_new_features_ignore_labels(self):
        s=self.sequence();p=np.array([105,68]);x=state_features(s,p)
        self.assertEqual(x.shape,(100,296));np.testing.assert_array_equal(x[:,:244],control_features(s,p))
        np.testing.assert_array_equal(x,state_features({**s,'y':np.full(100,2)},p))
    def test_team_swap_permutation_is_exact(self):
        s=self.sequence();p=np.array([105,68]);x=state_features(s,p)
        swapped=state_features({**s,'teams':1-s['teams']},p)
        np.testing.assert_array_equal(swap_state_columns(x),swapped)
        np.testing.assert_array_equal(swap_state_columns(swap_state_columns(x)),x)
    def test_no_ball_observations_produce_no_fake_interaction_anchors(self):
        x=state_features(self.sequence(),np.array([105,68]),np.ones(100,bool))
        self.assertTrue(np.isfinite(x).all());self.assertFalse(x[:,244:].any())
if __name__=='__main__':unittest.main()
