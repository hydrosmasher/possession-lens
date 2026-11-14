import unittest
try:
 import torch
 from controlswitch.focal_refinement import refinement_loss
except ImportError:
 torch=None

@unittest.skipIf(torch is None,'Optional PyTorch training dependency')
class FocalRefinementLossTests(unittest.TestCase):
 def test_focal_suppresses_easy_examples_and_has_finite_gradients(self):
  logits=torch.tensor([[[8.,-4.,-4.],[.2,.1,-2.]]],requires_grad=True);y=torch.tensor([[0,0]]);w=torch.ones((1,2));head=torch.zeros((1,2),requires_grad=True);t=torch.zeros((1,2));v=torch.zeros((1,2));ce=refinement_loss(logits,y,w,head,t,v,1.,'extra_ce');focal=refinement_loss(logits,y,w,head,t,v,1.,'focal2');self.assertLess(float(focal.detach()),float(ce.detach()));focal.backward();self.assertTrue(torch.isfinite(logits.grad).all())
 def test_unknown_only_has_zero_loss(self):
  logits=torch.zeros((1,3,3),requires_grad=True);y=torch.full((1,3),-1);z=torch.zeros((1,3));head=torch.zeros((1,3),requires_grad=True);loss=refinement_loss(logits,y,z,head,z,z,1.,'focal2');self.assertEqual(float(loss.detach()),0.)
if __name__=='__main__':unittest.main()
