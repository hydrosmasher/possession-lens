import unittest,numpy as np
try:
 import torch
 from controlswitch.ball_auxiliary import weighted_window,auxiliary_losses
 from controlswitch.gated_rnn import GatedPossession
except ImportError:
 torch=None

@unittest.skipIf(torch is None,'Optional PyTorch training dependency')
class BallAuxiliaryWindowTests(unittest.TestCase):
 def test_targets_never_enter_inputs_or_context_loss(self):
  n=20;x=np.zeros((n,508),np.float32);y=np.zeros(n,int);w=np.ones(n,np.float32);t=np.zeros(n,np.float32);v=np.ones(n,np.float32);ball=np.ones((n,2),np.float32);valid=np.ones(n,np.float32)
  a=weighted_window(x,y,w,t,v,0,ball,valid);b=weighted_window(x,y,w,t,v,0,ball*2,valid);np.testing.assert_array_equal(a[0],b[0]);self.assertTrue((a[6][:64]==0).all());self.assertTrue((a[6][84:]==0).all());self.assertEqual(float(a[6].sum()),20.)
 def test_complete_auxiliary_step_has_finite_gradients(self):
  torch.set_num_threads(1);torch.manual_seed(47);m=GatedPossession(layers=2,bidirectional=True,hidden=64,projection=64,input_features=508,boundary=True);aux=torch.nn.Linear(128,2);x=torch.zeros((2,40,508));y=torch.zeros((2,40),dtype=torch.long);w=torch.ones((2,40));t=torch.zeros((2,40));v=torch.ones((2,40));ball=torch.zeros((2,40,2));valid=torch.ones((2,40));loss,control,mse=auxiliary_losses(m,aux,x,y,w,t,v,ball,valid,10.);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(mse.detach()),0.);self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in list(m.parameters())+list(aux.parameters())))
if __name__=='__main__':unittest.main()
