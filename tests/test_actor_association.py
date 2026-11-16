import unittest,numpy as np,torch
from controlswitch.actor_association import owner_targets,augment_points,actor_logits,BASE_CONFIG
from controlswitch.player_set_state import NODES,FEATURES,transform_nodes
from controlswitch.player_set_rnn import PlayerSetPossession
class ActorAssociationTests(unittest.TestCase):
 def test_targets_require_unique_visible_exact_owner(self):
  nodes=np.zeros((4,2,NODES,FEATURES),np.float32);indices=np.full((4,2,NODES),-1,np.int32);owners=np.array([7,7,-1,7]);nodes[:,0,0,20]=1;nodes[:,0,0,10]=1;indices[:,0,0]=7;nodes[1,0,0,10]=0;nodes[3,1,0,20]=1;nodes[3,1,0,10]=1;indices[3,1,0]=7;np.testing.assert_array_equal(owner_targets(nodes,indices,owners),[0,-1,-1,-1])
 def test_augmentation_moves_owner_target_with_team(self):
  rng=np.random.default_rng(161);x=rng.normal(size=(8,2,NODES,FEATURES)).astype(np.float32);target=np.array([0,1,2,3,16,17,18,19]);codes=np.arange(8);a,b=augment_points(x,target,codes)
  for i,code in enumerate(codes):
   np.testing.assert_array_equal(a[i],transform_nodes(x[i],int(code)));self.assertEqual(b[i],(target[i]+NODES)%(2*NODES) if code&1 else target[i])
 def test_actor_scores_are_permutation_equivariant_and_differentiable(self):
  torch.manual_seed(161);m=PlayerSetPossession(BASE_CONFIG.copy(),pooling='global');x=torch.randn(3,2,NODES,FEATURES);x[...,20]=1;order=torch.randperm(NODES);a=actor_logits(m,x).reshape(3,2,NODES);b=actor_logits(m,x[:,:,order]).reshape(3,2,NODES);torch.testing.assert_close(b,a[:,:,order]);loss=torch.nn.functional.cross_entropy(a.flatten(1),torch.tensor([0,2,18]));loss.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.attention.parameters()));self.assertGreater(float(m.attention.weight.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()
