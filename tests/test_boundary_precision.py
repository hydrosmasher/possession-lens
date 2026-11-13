import unittest
import numpy as np
try:
 import torch
 from controlswitch.boundary_precision import blend_boundary,fitting_indices
except ImportError:torch=None
@unittest.skipIf(torch is None,'optional torch not installed')
class PrecisionTests(unittest.TestCase):
 def test_blend_extremes_and_bounds(self):
  a=np.array([.1,.9]);b=np.array([.8,.2]);np.testing.assert_allclose(blend_boundary(a,b,0),a);np.testing.assert_allclose(blend_boundary(a,b,1),b)
  with self.assertRaises(ValueError):blend_boundary(a,b,1.1)
  with self.assertRaises(ValueError):blend_boundary(a,np.array([float('nan'),0]),.5)
 def test_hard_samples_never_include_unknown_neighbourhoods(self):
  valid=np.array([False,True,False,True,False]);head=np.ones(5)
  ix=fitting_indices(valid,head);self.assertTrue(valid[ix].all());self.assertGreater(len(ix),0)
 def test_same_spell_audit_distinguishes_recovery_from_aggregate_counts(self):
  from controlswitch.boundary_precision import retention_counts
  y=np.r_[np.zeros(20,dtype=np.int8),np.ones(20,dtype=np.int8),np.zeros(20,dtype=np.int8)];current=np.zeros(60,dtype=np.int8)
  self.assertEqual(retention_counts(y,current,y,y)['lost_recovered_spells'],0)
  self.assertGreater(retention_counts(y,current,y,current)['lost_recovered_spells'],0)
