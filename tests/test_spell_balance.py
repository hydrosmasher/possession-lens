import unittest
import numpy as np
from controlswitch.rnn_spell_balance import spell_weights
from controlswitch.rnn import initialise,normalise,loss_gradients

class SpellBalanceTests(unittest.TestCase):
 def test_short_spells_get_more_weight_and_unknown_remains_zero(self):
    y=np.array([0]*4+[1]*16+[3]*2+[2]*6);w=spell_weights(y,8,1)
    self.assertEqual(w[0],2);self.assertEqual(w[5],.5);self.assertTrue((w[20:22]==0).all());self.assertTrue((w[-6:]==.5).all())
 def test_global_weighted_gradient_has_correct_scale(self):
    x=np.array([[.2,.4],[.4,.1],[.3,-.2]],np.float32);y=np.array([0,1,2]);m=initialise(x,hidden=3);z=normalise(m,x)
    loss,g,_=loss_gradients(m,z,y,weights=np.ones(3),normalise_weights=False)
    double,gg,_=loss_gradients(m,z,y,weights=np.full(3,2),normalise_weights=False)
    self.assertAlmostEqual(double,2*loss,places=6)
    for k in g:np.testing.assert_allclose(gg[k],2*g[k],atol=1e-7)
    original=m['wh'][1,2];eps=.002
    m['wh'][1,2]=original+eps;plus=loss_gradients(m,z,y,weights=np.array([2,1,3]),normalise_weights=False)[0]
    m['wh'][1,2]=original-eps;minus=loss_gradients(m,z,y,weights=np.array([2,1,3]),normalise_weights=False)[0];m['wh'][1,2]=original
    _,grad,_=loss_gradients(m,z,y,weights=np.array([2,1,3]),normalise_weights=False)
    self.assertAlmostEqual(float(grad['wh'][1,2]),(plus-minus)/(2*eps),delta=.0003)
