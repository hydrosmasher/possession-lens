import unittest,tempfile,importlib.util
from pathlib import Path
import numpy as np
HAVE_TORCH=importlib.util.find_spec('torch') is not None
if HAVE_TORCH:
 import torch
 from controlswitch.gated_rnn import GatedPossession,make_window,predict,save_checkpoint,CORE,CONTEXT,scaling
 from controlswitch.gated_experiment import masked_loss
 from controlswitch.rnn_runtime import load_frozen,infer_frozen

@unittest.skipUnless(HAVE_TORCH,'Optional requirements-gated.txt not installed')
class GatedTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(42)
 def test_core_mask_and_padding_cannot_supply_label_loss(self):
  x=np.ones((300,130),np.float32);y=np.zeros(300,np.int64);xx,yy,t,v=make_window(x,y,0)
  self.assertTrue((yy[:CONTEXT]==-1).all());self.assertTrue((xx[:CONTEXT]==0).all());self.assertTrue((yy[CONTEXT:CONTEXT+CORE]==0).all());self.assertTrue((yy[CONTEXT+CORE:]==-1).all())
  xx,yy,t,v=make_window(x,y,256);self.assertEqual(np.sum(yy>=0),44);self.assertTrue((xx[CONTEXT+44:]==0).all())
 def test_architectures_have_finite_gradients_and_clipping_bounds(self):
  for cell in ('gru','lstm'):
   m=GatedPossession(cell=cell,layers=2,bidirectional=True,boundary=True);x=torch.randn(2,16,130);y=torch.zeros(2,16,dtype=torch.long);logits,b=m(x)
   loss=masked_loss(logits,y,b,torch.zeros(2,16),torch.ones(2,16));(loss*100).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
   self.assertLessEqual(float(torch.stack([p.grad.norm()**2 for p in m.parameters() if p.grad is not None]).sum().sqrt()),1.00001)
 def test_unknown_targets_are_loss_invariant(self):
  m=GatedPossession();m.eval();logits,_=m(torch.randn(1,8,130));y=torch.tensor([[0,1,-1,-1,0,1,-1,-1]]);a=masked_loss(logits,y);other=logits.detach().clone();other[:,[2,3,6,7]]=100
  self.assertAlmostEqual(float(a.detach()),float(masked_loss(other,y)),places=6)
 def test_gate_initialisation_and_forget_bias(self):
  for cell,gates in [('gru',3),('lstm',4)]:
   m=GatedPossession(cell=cell,layers=2,bidirectional=True)
   for name,p in m.recurrent.named_parameters():
    if 'weight_hh' in name:
     for gate in p.chunk(gates,0):np.testing.assert_allclose((gate.T@gate).detach().numpy(),np.eye(32),atol=1e-5)
    if cell=='lstm' and 'bias_ih' in name:np.testing.assert_array_equal(p.detach().numpy()[32:64],1)
 def test_frozen_roundtrip_batching_and_feature_label_independence(self):
  rng=np.random.default_rng(3);n=350;s={'xy':rng.normal(size=(n,4,2)).astype(np.float32),'speed':np.zeros((n,4),np.float32),'ball':rng.normal(size=(n,2)).astype(np.float32),'ball_speed':np.zeros(n,np.float32),'teams':np.array([0,0,1,1]),'y':np.zeros(n,np.int8)};pitch=np.array([105,68]);m=GatedPossession(layers=2,bidirectional=True,boundary=True);mean=np.zeros(130,np.float32);scale=np.ones(130,np.float32)
  x=rng.normal(size=(n,130)).astype(np.float32);np.testing.assert_allclose(predict(m,x,mean,scale,batch=1)[0],predict(m,x,mean,scale,batch=32)[0],atol=1e-6)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'model.npz';save_checkpoint(p,m,mean,scale,36,.5);f=load_frozen(p);a=infer_frozen(f,s,pitch);s['y'][:]=3;b=infer_frozen(f,s,pitch);np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_allclose(a[1],b[1],atol=1e-6)
   with np.load(p,allow_pickle=False) as d:payload={k:d[k] for k in d.files}
   payload['scale'][0]=0;np.savez(p,**payload)
   with self.assertRaises(ValueError):load_frozen(p)
