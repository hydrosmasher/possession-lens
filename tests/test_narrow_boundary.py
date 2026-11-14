import unittest,numpy as np
from controlswitch.narrow_boundary import narrow_targets
from controlswitch.ranking_experiment import SPECS

class NarrowBoundaryTargetTests(unittest.TestCase):
 def test_targets_localise_turnover_and_exclude_dead_or_unknown(self):
  y=np.r_[np.zeros(30,int),np.ones(30,int)];a,v=narrow_targets(y,2);b,_=narrow_targets(y,5)
  self.assertEqual(int(a.sum()),5);self.assertEqual(int(b.sum()),11);self.assertTrue(v.all())
  for pause in (2,3):
   y=np.r_[np.zeros(30,int),np.full(30,pause),np.ones(30,int)];t,v=narrow_targets(y,2);self.assertFalse(t.any());self.assertFalse(v[30:60].any())
 def test_original030_experiment_does_not_expand_with_runtime_support(self):
  self.assertEqual(len(SPECS),10);self.assertNotIn('joint_motion_quality1_boundary_full',{s['name'] for s in SPECS})
if __name__=='__main__':unittest.main()
