import unittest
import numpy as np
from controlswitch import rnn_deep as r
class DeepTests(unittest.TestCase):
 def test_stacked_bptt_and_l2_gradients(self):
  x=np.array([[[.2,.4],[.4,.1],[.3,-.2]]],np.float32);y=np.array([[0,1,2]]);m=r.initialise(x[0],(3,2));z=r.normalise(m,x);loss,g=r.loss_gradients(m,z,y,l2=.01)
  for k,i in [('wx0',(0,1)),('wh0',(1,2)),('wx1',(1,0)),('wh1',(0,1)),('b0',(0,)),('b1',(0,)),('wo',(1,0)),('bo',(2,))]:
   v=m[k][i];e=.002;m[k][i]=v+e;p=r.loss_gradients(m,z,y,l2=.01)[0];m[k][i]=v-e;n=r.loss_gradients(m,z,y,l2=.01)[0];m[k][i]=v
   self.assertAlmostEqual(float(g[k][i]),(p-n)/(2*e),delta=.0003)
 def test_chunked_inference_preserves_all_layer_states(self):
  x=np.random.default_rng(3).normal(size=(25,3)).astype(np.float32);m=r.initialise(x,(4,3,2));np.testing.assert_allclose(r.predict(m,x,3),r.predict(m,x,100),atol=1e-6)
 def test_dropout_reproducible_and_training_normalisation_only(self):
  x=np.arange(12,dtype=np.float32).reshape(6,2)/12;m=r.initialise(x,(3,3));y=np.array([[0,1,2,0,1,2]])
  a=r.loss_gradients(m,r.normalise(m,x[None]),y,.2,np.random.default_rng(2));b=r.loss_gradients(m,r.normalise(m,x[None]),y,.2,np.random.default_rng(2));self.assertEqual(a[0],b[0]);self.assertTrue(all(np.isfinite(v).all() for v in a[1].values()))
 def test_frozen_stacked_checkpoint_roundtrip_and_bad_shapes_rejected(self):
  import tempfile
  from pathlib import Path
  from controlswitch.rnn_runtime import load_frozen
  x=np.ones((5,2),np.float32);m=r.initialise(x,(3,2))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'m.npz';np.savez_compressed(p,**m,version=np.array(r.VERSION),symmetric=np.array(True),transition_penalty=np.array(24.));f=load_frozen(p);np.testing.assert_allclose(r.predict(f['model'],x),r.predict(m,x))
   m['wx1']=np.zeros((9,2));np.savez_compressed(p,**m,version=np.array(r.VERSION),symmetric=np.array(True),transition_penalty=np.array(24.))
   with self.assertRaises(ValueError):load_frozen(p)
