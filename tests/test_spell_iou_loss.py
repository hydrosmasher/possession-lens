import unittest,numpy as np,torch
from controlswitch.spell_iou_loss import local_spell_iou
class LocalSpellIoUTests(unittest.TestCase):
 def sample(self):
  y=torch.tensor([[-1,1,1,0,0,1,1,-1]]);logits=torch.full((1,8,3),-8.);logits[:,:,2]=-10
  for i,k in enumerate(y[0]):
   if k>=0:logits[0,i,k]=8
  return logits,y
 def test_correct_spell_beats_shifted_and_wrong_team(self):
  p,y=self.sample();good=float(local_spell_iou(p,y));shift=p.clone();shift[0,2,:2]=torch.tensor([8.,-8.]);bad=p.clone();bad[0,3:5,:2]=torch.tensor([-8.,8.]);self.assertLess(good,1e-5);self.assertGreater(float(local_spell_iou(shift,y)),good+.2);self.assertGreater(float(local_spell_iou(bad,y)),.9)
 def test_unknown_padding_is_excluded_and_gradients_are_finite(self):
  p,y=self.sample();p=torch.randn_like(p,requires_grad=True);loss=local_spell_iou(p,y);loss.backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
  padded=torch.nn.functional.pad(p.detach(),(0,0,7,5));py=torch.nn.functional.pad(y,(7,5),value=-1);np.testing.assert_allclose(float(local_spell_iou(padded,py)),float(loss.detach()),atol=1e-6)
 def test_no_complete_live_spells_has_zero_loss(self):
  p=torch.randn(2,30,3,requires_grad=True);y=torch.zeros(2,30,dtype=torch.long);loss=local_spell_iou(p,y);loss.backward();self.assertEqual(float(loss.detach()),0);self.assertTrue((p.grad==0).all())
if __name__=='__main__':unittest.main()
