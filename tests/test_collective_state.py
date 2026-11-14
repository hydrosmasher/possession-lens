import unittest
import numpy as np
from controlswitch.collective_state import collective_features,collective_state,swap_collective,attack_directions

class CollectiveStateTests(unittest.TestCase):
 def sequence(self):
  rng=np.random.default_rng(38);n=100;xy=rng.normal(size=(n,4,2));xy[:,:2,0]-=30;xy[:,2:,0]+=30
  return {'xy':xy,'teams':np.array([0,0,1,1]),'speed':np.ones((n,4)),'ball':np.zeros((n,2)),'ball_speed':np.zeros(n)}
 def test_team_swap_and_ball_independence(self):
  s=self.sequence();pitch=np.array([105,68]);a=collective_state(s,pitch);b=collective_state({**s,'teams':1-s['teams']},pitch);self.assertEqual(a.shape,(100,504));np.testing.assert_array_equal(swap_collective(a),b)
  x=collective_features(s,pitch);np.testing.assert_array_equal(x,collective_features({**s,'ball':np.full((100,2),np.nan)},pitch));self.assertTrue(np.isfinite(x).all());self.assertTrue(np.any(x!=0))
 def test_attack_direction_is_equivariant_to_field_side_and_team_swap(self):
  s=self.sequence();np.testing.assert_array_equal(attack_directions(s['xy'],s['teams']),[1,-1]);np.testing.assert_array_equal(attack_directions(s['xy'],1-s['teams']),[-1,1]);xy=s['xy'].copy();xy[:,:,0]*=-1;np.testing.assert_array_equal(attack_directions(xy,s['teams']),[-1,1])
if __name__=='__main__':unittest.main()
