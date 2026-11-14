import unittest
import numpy as np
try:
 import torch
 from controlswitch.spell_gru import weighted_window,loss_value
except ImportError:
 torch=None

@unittest.skipIf(torch is None,'Optional PyTorch training dependency')
class SpellGRULossTests(unittest.TestCase):
 def test_window_excludes_context_padding_and_unknown_from_weight(self):
  x=np.zeros((20,130),np.float32);y=np.r_[np.zeros(10,int),np.full(10,-1)];w=np.ones(20,np.float32);t=np.zeros(20,np.float32);v=np.ones(20,np.float32)
  xx,yy,ww,tt,vv=weighted_window(x,y,w,t,v,0)
  self.assertEqual(int((ww>0).sum()),10);self.assertTrue((ww[yy<0]==0).all());self.assertTrue((vv[:64]==0).all());self.assertTrue((vv[84:]==0).all())
 def test_no_known_supervision_has_zero_loss_and_finite_gradient(self):
  logits=torch.zeros((1,5,3),requires_grad=True);head=torch.zeros((1,5),requires_grad=True);y=torch.full((1,5),-1);z=torch.zeros((1,5))
  loss=loss_value(logits,y,z,head,z,z,1.);loss.backward()
  self.assertEqual(float(loss.detach()),0.);self.assertTrue(torch.isfinite(logits.grad).all());self.assertTrue(torch.isfinite(head.grad).all())
if __name__=='__main__':unittest.main()
