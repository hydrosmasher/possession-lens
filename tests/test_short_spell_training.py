import unittest,numpy as np
from controlswitch.short_spell_training import short_spell_mask,training_anchor
class ShortTrainingTests(unittest.TestCase):
 def test_unknown_dead_and_team_boundaries_separate_spells(self):
  y=np.array([0,0,-1,0,1,1,2,1,1,1]);mask,spells=short_spell_mask(y,2)
  self.assertEqual(spells,[(0,2),(3,4),(4,6)])
  np.testing.assert_array_equal(mask,[1,1,0,1,1,1,0,0,0,0])
 def test_threshold_inclusive_and_half_end_preserved(self):
  y=np.r_[np.zeros(75,int),np.ones(76,int),[-1,1]];mask,spells=short_spell_mask(y)
  self.assertEqual(spells,[(0,75),(152,153)]);self.assertEqual(mask.sum(),76)
 def test_sampler_only_uses_known_training_frames(self):
  y=np.r_[np.zeros(10,int),np.full(20,-1),np.ones(300,int)];_,s=short_spell_mask(y);e={'known':np.flatnonzero(y>=0),'short_spells':s};rng=np.random.default_rng(153);anchors=np.array([training_anchor(e,rng,True) for _ in range(2000)])
  self.assertTrue((y[anchors]>=0).all());self.assertGreater(np.mean(anchors<10),.45)
  rng=np.random.default_rng(153);normal=np.array([training_anchor(e,rng,False) for _ in range(2000)]);self.assertLess(np.mean(normal<10),.07)
if __name__=='__main__':unittest.main()
