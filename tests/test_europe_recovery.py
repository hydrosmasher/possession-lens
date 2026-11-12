import unittest,tempfile
from pathlib import Path
import numpy as np
from controlswitch import rnn_deep as net
from controlswitch.europe_recovery import loss_weights,boundary_supervision,accepted,save_model
from controlswitch.rnn_runtime import load_frozen,infer_frozen
from controlswitch.rnn_boundaries import invariant_boundary_features

class RecoveryTests(unittest.TestCase):
 def test_weighted_gradient_and_unknowns(self):
  x=np.array([[[.1,.3],[.4,.2],[.2,.8]]],np.float32);y=np.array([[0,1,-1]]);w=np.array([[1.,3.,99.]],np.float32);m=net.initialise(x[0],(3,))
  loss,g=net.loss_gradients(m,x,y,sample_weight=w)
  w[0,-1]=0;self.assertEqual(loss,net.loss_gradients(m,x,y,sample_weight=w)[0])
  for k,ix in [('wx0',(0,1)),('wh0',(1,2)),('wo',(1,0)),('bo',(1,))]:
   v=m[k][ix];e=.002;m[k][ix]=v+e;a=net.loss_gradients(m,x,y,sample_weight=w)[0];m[k][ix]=v-e;b=net.loss_gradients(m,x,y,sample_weight=w)[0];m[k][ix]=v
   self.assertAlmostEqual(float(g[k][ix]),(a-b)/(2*e),delta=.0003)
  for bad in [np.ones((2,3)),np.array([[1.,-1.,0.]])]:
   with self.assertRaises(ValueError):net.loss_gradients(m,x,y,sample_weight=bad)
 def test_short_uncensored_spells_upweighted_unknowns_excluded(self):
  y=np.r_[np.zeros(1000),np.ones(20),np.zeros(500),[3],np.ones(10)].astype(np.int8);w=loss_weights(y)
  self.assertGreater(w[1010],w[1300]);self.assertEqual(w[1520],0);self.assertEqual(w[-1],1)
 def test_unknown_boundaries_not_positive_supervision(self):
  y=np.r_[np.zeros(100),np.full(20,3),np.ones(100)].astype(np.int8);target,valid=boundary_supervision(y)
  self.assertEqual(int(target.sum()),0);self.assertFalse(valid[75:145].any())
 def test_stacked_boundary_checkpoint_roundtrip(self):
  n=60;s={'xy':np.zeros((n,2,2),np.float32),'speed':np.zeros((n,2),np.float32),'ball':np.zeros((n,2),np.float32),'ball_speed':np.zeros(n,np.float32),'teams':np.array([0,1])};pitch=np.array([105,68])
  from controlswitch.model import instantaneous_features,decode,probabilities
  from controlswitch.rnn_capacity import observation_probabilities
  model=net.initialise(instantaneous_features(s,pitch),(3,));p=observation_probabilities(model,s,pitch);x=invariant_boundary_features(s,pitch,p);head={'mean':np.zeros(x.shape[1],np.float32),'scale':np.ones(x.shape[1],np.float32),'weights':np.zeros((x.shape[1]+1,2),np.float32)}
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'m.npz';save_model(path,model,12,head,.5);f=load_frozen(path);pred,_=infer_frozen(f,s,pitch)
   np.testing.assert_array_equal(pred,decode(p,12,probabilities(head,x)[:,1],.5))
 def test_reserved_import_verifies_bytes_without_parsing_labels(self):
  import hashlib,zipfile,json
  from unittest.mock import patch
  from controlswitch.skillcorner_fresh_import import import_fresh_test
  payload=b'not-json: labels must not be parsed during import'
  size=len(payload);sha=hashlib.sha1(f'blob {size}\0'.encode()+payload).hexdigest()
  catalog={'matches':[{'match_id':'2417','files':[{'name':'structured_data.json','size':size,'sha':sha}]}]}
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);archive=p/'source.zip'
   with zipfile.ZipFile(archive,'w') as z:z.writestr('release/data/matches/2417/structured_data.json',payload);z.writestr('release/LICENSE','MIT test fixture')
   with patch('controlswitch.skillcorner_fresh_import.FRESH_CATALOG',catalog):
    m=import_fresh_test(archive,p/'raw');self.assertFalse(m['held_out_contents_parsed']);self.assertEqual(m['split']['test'],['2417'])
    self.assertEqual((p/'raw/2417/structured_data.json').read_bytes(),payload)
    with self.assertRaises(ValueError):import_fresh_test(archive,p/'raw')
