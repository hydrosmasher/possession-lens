import unittest,numpy as np
from controlswitch.robust_dead_state import robust_dead_features,blend_dead
class RobustDeadTests(unittest.TestCase):
 def test_team_invariance_missingness_and_no_label_inputs(self):
  rng=np.random.default_rng(59);n=70;s={'xy':np.cumsum(rng.normal(0,.04,(n,6,2)),axis=0),'ball':np.cumsum(rng.normal(0,.05,(n,2)),axis=0),'teams':np.array([0,0,0,1,1,1]),'speed':np.ones((n,6)),'ball_speed':np.ones(n)};s['xy'][20:35]=np.nan;s['ball'][40:60]=np.nan;pitch=np.array([105,68]);x=robust_dead_features(s,pitch);self.assertEqual(x.shape,(n,487));self.assertTrue(np.isfinite(x).all());np.testing.assert_allclose(x,robust_dead_features({**s,'teams':1-s['teams']},pitch),atol=1e-6);np.testing.assert_array_equal(x,robust_dead_features({**s,'y':np.ones(n)},pitch))
 def test_fusion_preserves_conditional_team_control(self):
  q=np.array([[.6,.2,.2],[.1,.4,.5]]);d=np.array([.8,.1]);p=blend_dead(q,d,.5);np.testing.assert_allclose(p.sum(1),1);np.testing.assert_allclose(p[:,0]/p[:,1],q[:,0]/q[:,1]);self.assertTrue(((p>0)&(p<1)).all())
  with self.assertRaises(ValueError):blend_dead(q,d,.7)
if __name__=='__main__':unittest.main()
