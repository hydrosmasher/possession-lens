import unittest,numpy as np
from controlswitch.joint_state import joint_features,swap_joint
from controlswitch.relative_state import relative_features

class JointStateTests(unittest.TestCase):
 def test_exact_team_swap_and_preserved_relative_prefix(self):
  rng=np.random.default_rng(40);n=100;s={'xy':rng.normal(size=(n,4,2)),'speed':np.ones((n,4)),'teams':np.array([0,0,1,1]),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)};pitch=np.array([105,68]);x=joint_features(s,pitch);y=joint_features({**s,'teams':1-s['teams'],'y':np.ones(n)},pitch)
  self.assertEqual(x.shape,(n,508));np.testing.assert_array_equal(x[:,:416],relative_features(s,pitch));np.testing.assert_array_equal(swap_joint(x),y)
if __name__=='__main__':unittest.main()
