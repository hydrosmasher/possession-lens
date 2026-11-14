import unittest
import numpy as np
from controlswitch.relative_state import motion_features,relative_features,swap_relative
from controlswitch.control_state import state_features

class RelativeMotionFeaturesTests(unittest.TestCase):
 def sequence(self):
  rng=np.random.default_rng(37);n=100
  return {'xy':rng.normal(size=(n,4,2)),'speed':np.ones((n,4)),'teams':np.array([0,0,1,1]),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)}
 def test_exact_team_swap_and_old_prefix(self):
  s=self.sequence();pitch=np.array([105,68]);a=relative_features(s,pitch);b=relative_features({**s,'teams':1-s['teams'],'y':np.full(100,2)},pitch)
  self.assertEqual(a.shape,(100,416));np.testing.assert_array_equal(a[:,:296],state_features(s,pitch));np.testing.assert_array_equal(swap_relative(a),b)
 def test_stationary_ball_has_no_heading_turn_or_impulse(self):
  s=self.sequence();s['ball'][:]=0;s['xy'][:]=1;x=motion_features(s)
  np.testing.assert_array_equal(x[:,[4,5,16,17]],np.zeros((100,4)))
 def test_missing_ball_produces_no_contact_or_impulse(self):
  s=self.sequence();s['ball'][:]=np.nan;x=motion_features(s)
  self.assertTrue(np.isfinite(x).all());np.testing.assert_array_equal(x[:,[3,4,5,7,8,9,10,11,15,16,17,19,20,21,22,23]],np.zeros((100,16)))
if __name__=='__main__':unittest.main()
