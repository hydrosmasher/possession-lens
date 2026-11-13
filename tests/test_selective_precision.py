import unittest
import numpy as np
try:
 import torch
 from controlswitch.selective_precision import protected_boundary
except ImportError:torch=None
@unittest.skipIf(torch is None,'optional torch not installed')
class SelectiveTests(unittest.TestCase):
 def test_strong_emission_change_preserves_original_head(self):
  p=np.r_[np.tile([.9,.1,0.],(30,1)),np.tile([.1,.9,0.],(30,1))];head=np.full(60,.8);q=np.zeros(60)
  result=protected_boundary(p,head,q,.5,.5);self.assertAlmostEqual(float(result[30]),.8,places=6);self.assertAlmostEqual(float(result[0]),.4,places=6)
 def test_team_swap_preserves_gate(self):
  rng=np.random.default_rng(2);p=rng.dirichlet([1,1,1],100);head=np.full(100,.7);q=np.full(100,.2)
  np.testing.assert_allclose(protected_boundary(p,head,q,.5,.25),protected_boundary(p[:,[1,0,2]],head,q,.5,.25))
