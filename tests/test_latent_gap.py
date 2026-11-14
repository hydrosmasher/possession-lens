import unittest,numpy as np
from controlswitch.latent_gap import ball_hypothesis,latent_features,swap_latent

class LatentBallHypothesisTests(unittest.TestCase):
 def test_bracketed_hypothesis_never_changes_original_or_fills_ends(self):
  b=np.full((100,2),np.nan);b[10]=[0,0];b[90]=[20,0];s={'ball':b};h,seen,filled=ball_hypothesis(s)
  self.assertTrue(filled[11:90].all());self.assertFalse(filled[:11].any());self.assertFalse(filled[90:].any());self.assertTrue(np.isnan(s['ball'][20]).all());np.testing.assert_array_equal(h[[10,90]],b[[10,90]])
 def test_implausible_or_unbracketed_hypotheses_are_rejected(self):
  b=np.full((400,2),np.nan);b[0]=[0,0];b[-1]=[10,0];self.assertFalse(ball_hypothesis({'ball':b})[2].any());b=np.array([[0.,0.],[np.nan,np.nan],[100.,0.]]);self.assertFalse(ball_hypothesis({'ball':b})[2].any())
 def test_team_swap_preserves_identity_free_geometry(self):
  rng=np.random.default_rng(44);n=100;s={'xy':rng.normal(size=(n,4,2)),'teams':np.array([0,0,1,1]),'speed':np.ones((n,4)),'ball':np.column_stack([np.linspace(0,10,n),np.zeros(n)]),'ball_speed':np.ones(n)};s['ball'][20:80]=np.nan;pitch=np.array([105,68]);a=latent_features(s,pitch);b=latent_features({**s,'teams':1-s['teams']},pitch);self.assertEqual(a.shape,(n,512));np.testing.assert_array_equal(swap_latent(a),b)
if __name__=='__main__':unittest.main()
