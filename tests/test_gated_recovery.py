import unittest
import numpy as np
try:
 import torch
 from controlswitch.gated_recovery import transition_indices,sample_start,weighted_ce
except ImportError:torch=None

@unittest.skipIf(torch is None,'optional torch not installed')
class RecoveryTests(unittest.TestCase):
 def test_events_exclude_unknown_gaps(self):
  np.testing.assert_array_equal(transition_indices([0,0,-1,1,1,0,3,1]),[5])
 def test_weighted_loss_excludes_unknown(self):
  x=torch.tensor([[[1.,0.,0.],[0.,2.,0.]]],requires_grad=True);y=torch.tensor([[0,-1]])
  value=weighted_ce(x,y,torch.tensor([[2.,100.]]));value.backward()
  self.assertAlmostEqual(float(value.detach()),float(torch.nn.functional.cross_entropy(x[0,:1],y[0,:1]).detach()),places=6)
  self.assertEqual(float(x.grad[0,1].abs().sum()),0)
 def test_short_sequence_sampling_in_bounds(self):
  rng=np.random.default_rng(3)
  for _ in range(100):self.assertIn(sample_start(np.array([0,1,0]),rng,True),range(3))
