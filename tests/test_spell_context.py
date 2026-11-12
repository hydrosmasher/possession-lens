import unittest
import numpy as np
from controlswitch.europe_recovery import loss_weights
from controlswitch import rnn_deep as net
class SpellContextTests(unittest.TestCase):
 def test_unknown_labels_zero_weight_and_censored_runs_not_duration_boosted(self):
  y=np.r_[np.full(50,3),np.zeros(100),np.full(50,3)].astype(np.int8)
  w=loss_weights(y)
  np.testing.assert_array_equal(w[y==3],0)
  self.assertTrue(np.all(w[y==0]>=1))
  self.assertTrue(np.all(w[y==0]<=1.5))
 def test_weighted_two_layer_gradient_and_unknown_label_invariance(self):
  x=np.random.default_rng(4).normal(size=(1,5,2)).astype(np.float32);y=np.array([[0,1,-1,1,0]]);w=np.array([[1,3,0,3,1]],np.float32);m=net.initialise(x[0],(3,2));z=net.normalise(m,x)
  loss,g=net.loss_gradients(m,z,y,sample_weight=w)
  for k,i in [('wx0',(0,1)),('wx1',(1,0)),('wo',(1,0))]:
   v=m[k][i];eps=.002;m[k][i]=v+eps;plus=net.loss_gradients(m,z,y,sample_weight=w)[0];m[k][i]=v-eps;minus=net.loss_gradients(m,z,y,sample_weight=w)[0];m[k][i]=v
   self.assertAlmostEqual(float(g[k][i]),(plus-minus)/(2*eps),delta=.0003)
  other=y.copy();other[0,2]=2
  self.assertAlmostEqual(loss,net.loss_gradients(m,z,other,sample_weight=w)[0],places=6)
