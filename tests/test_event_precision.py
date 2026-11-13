import unittest
import numpy as np
try:
 import torch
 from controlswitch.event_precision import proposals,label_event,apply_edits,role_features,event_features
except ImportError:torch=None
@unittest.skipIf(torch is None,'optional torch not installed')
class EventTests(unittest.TestCase):
 def setUp(self):self.pred=np.r_[np.zeros(40,dtype=np.int8),np.ones(20,dtype=np.int8),np.zeros(40,dtype=np.int8)]
 def test_proposals_and_conservative_edit(self):
  events=proposals(self.pred);self.assertEqual(len(events),1);p,n=apply_edits(self.pred,events,[.05],.1,125);self.assertEqual(n,1);self.assertTrue((p==0).all());p,n=apply_edits(self.pred,events,[.5],.1,125);np.testing.assert_array_equal(p,self.pred)
 def test_unknown_neighbourhoods_never_labelled(self):
  e=proposals(self.pred)[0];y=self.pred.copy();self.assertEqual(label_event(y,e),1);y[30]=3;self.assertIsNone(label_event(y,e));self.assertEqual(label_event(np.zeros(100,dtype=np.int8),e),0)
 def test_role_features_are_team_swap_invariant(self):
  rng=np.random.default_rng(7);base=rng.normal(size=(100,30));inter=rng.normal(size=(100,10));swapped=np.c_[base[:,:6],base[:,18:30],base[:,6:18]];si=np.c_[inter[:,5:],inter[:,:5]]
  np.testing.assert_allclose(role_features(base,inter,1),role_features(swapped,si,0))
 def test_no_cascading_recompute_or_unrelated_boundary_change(self):
  pred=np.r_[np.zeros(40,dtype=np.int8),np.ones(20,dtype=np.int8),np.zeros(20,dtype=np.int8),np.ones(40,dtype=np.int8)];events=proposals(pred);p,n=apply_edits(pred,events,[.01,.02],.1,125);self.assertEqual(n,1);np.testing.assert_array_equal(p[80:],pred[80:])

 def test_full_event_features_are_team_swap_invariant(self):
  rng=np.random.default_rng(10);base=rng.normal(size=(100,30));inter=rng.normal(size=(100,10));prob=rng.dirichlet([1,1,1],100);head=rng.random(100);e=proposals(self.pred)[0]
  swapped_base=np.c_[base[:,:6],base[:,18:30],base[:,6:18]];swapped_inter=np.c_[inter[:,5:],inter[:,:5]];roles=[role_features(base,inter,t) for t in (0,1)];sr=[role_features(swapped_base,swapped_inter,t) for t in (0,1)];se=(e[0],e[1],1-e[2],1-e[3],e[4],e[5])
  np.testing.assert_allclose(event_features(roles,prob,head,e),event_features(sr,prob[:,[1,0,2]],head,se))
 def test_compact_features_have_twenty_finite_values(self):
  from controlswitch.event_precision import compact_features
  x=np.arange(340,dtype=np.float32).reshape(2,170);out=compact_features(x);self.assertEqual(out.shape,(2,20));self.assertTrue(np.isfinite(out).all())

 def test_unknown_or_stop_predictions_are_not_edit_candidates(self):
  for code in (-1,2,3):
   pred=self.pred.copy();pred[:40]=code;pred[60:]=code;self.assertEqual(proposals(pred),[])

class ClassifierRegularizationTests(unittest.TestCase):
 def test_regularisation_parameter_rejects_invalid_values_and_shrinks_weights(self):
  from controlswitch.model import fit_classifier
  x=np.r_[np.full((20,2),-1.),np.full((20,2),1.)];y=np.r_[np.zeros(20,dtype=int),np.ones(20,dtype=int)]
  with self.assertRaises(ValueError):fit_classifier(x,y,classes=2,l2=-1)
  weak=fit_classifier(x,y,classes=2,epochs=45,l2=.0005);strong=fit_classifier(x,y,classes=2,epochs=45,l2=1.)
  self.assertLess(np.linalg.norm(strong['weights'][:-1]),np.linalg.norm(weak['weights'][:-1]))
