import unittest,tempfile
from pathlib import Path
import numpy as np,torch
from controlswitch.player_set_state import player_nodes,transform_nodes,NODES,FEATURES
from controlswitch.player_set_rnn import warm_start,save_checkpoint,load_checkpoint
from controlswitch.gated_rnn import load_checkpoint as load_old
from controlswitch.spatial_symmetry import reflect_sequence
class PlayerSetTests(unittest.TestCase):
 def sequence(self):
  rng=np.random.default_rng(60);n=64;return {'xy':np.cumsum(rng.normal(0,.05,(n,6,2)),axis=0),'teams':np.array([0,0,0,1,1,1]),'speed':np.ones((n,6)),'ball':np.cumsum(rng.normal(0,.07,(n,2)),axis=0),'ball_speed':np.ones(n)}
 def test_nodes_translation_reflection_swap_and_missingness(self):
  s=self.sequence();pitch=np.array([105,68]);x=player_nodes(s,pitch);self.assertEqual(x.shape,(64,2,NODES,FEATURES));shift={**s,'xy':s['xy']+np.array([10,5],np.float32),'ball':s['ball']+np.array([10,5],np.float32)};np.testing.assert_allclose(player_nodes(shift,pitch),x,atol=2e-5)
  for code,fx,fy in ((2,True,False),(4,False,True),(6,True,True)):np.testing.assert_allclose(player_nodes(reflect_sequence(s,fx,fy),pitch),transform_nodes(x,code),atol=1e-6)
  np.testing.assert_allclose(player_nodes({**s,'teams':1-s['teams']},pitch),transform_nodes(x,1),atol=1e-6);s['xy'][10:25]=np.nan;s['ball'][35:50]=np.nan;y=player_nodes(s,pitch);self.assertTrue(np.isfinite(y).all());self.assertTrue((y[10:25]==0).all());np.testing.assert_array_equal(y,player_nodes({**s,'y':np.ones(64)},pitch))
 def old(self):return load_old(Path(__file__).resolve().parents[1]/'models/archive/gru-short053-validation-only.npz')
 def test_zero_adapter_starts_at_original_outputs_and_roundtrips(self):
  old=self.old();m=warm_start(old);m.eval();torch.set_num_threads(1);torch.manual_seed(60);x=torch.randn(2,20,508);nodes=torch.randn(2,20,2,NODES,FEATURES);nodes[...,20]=1
  with torch.inference_mode():a,ah=old['model'](x);b,bh=m(x,nodes)
  np.testing.assert_allclose(a,b,atol=1e-6);np.testing.assert_allclose(ah,bh,atol=1e-6)
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'m.npz';save_checkpoint(p,m,old['mean'],old['scale']);new=load_checkpoint(p)
   with torch.inference_mode():c,ch=new['model'](x,nodes)
   np.testing.assert_array_equal(b,c);np.testing.assert_array_equal(bh,ch)
 def test_learned_pool_is_player_order_invariant_and_has_finite_gradients(self):
  old=self.old();m=warm_start(old);m.eval();torch.manual_seed(60);torch.nn.init.normal_(m.node_projection.weight,std=.05);x=torch.randn(2,16,508);nodes=torch.randn(2,16,2,NODES,FEATURES);nodes[...,20]=1;order=torch.randperm(NODES);a,ah=m(x,nodes);b,bh=m(x,nodes[:,:,:,order]);np.testing.assert_allclose(a.detach(),b.detach(),atol=2e-6);loss=a.square().mean();loss.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None));self.assertGreater(float(m.encoder[0].weight.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()
