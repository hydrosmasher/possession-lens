import unittest,numpy as np
from controlswitch.contact_state import contact_features,contact_state,swap_contact
from controlswitch.spatial_symmetry import reflect_sequence
class ContactStateTests(unittest.TestCase):
 def sequence(self):
  rng=np.random.default_rng(57);n=95;return {'xy':np.cumsum(rng.normal(0,.04,(n,6,2)),axis=0),'ball':np.cumsum(rng.normal(0,.06,(n,2)),axis=0),'teams':np.array([0,0,0,1,1,1]),'speed':np.ones((n,6)),'ball_speed':np.ones(n)}
 def test_finite_when_ball_players_are_missing_and_labels_are_ignored(self):
  s=self.sequence();s['ball'][10:30]=np.nan;s['xy'][40:55]=np.nan;x=contact_features(s);self.assertEqual(x.shape,(95,64));self.assertTrue(np.isfinite(x).all());np.testing.assert_array_equal(x,contact_features({**s,'y':np.ones(95)}));self.assertTrue((x[10:30]==0).all())
 def test_team_swap_reflection_and_player_order(self):
  s=self.sequence();x=contact_features(s);sw=contact_features({**s,'teams':1-s['teams']});np.testing.assert_allclose(sw,np.column_stack([x[:,32:],x[:,:32]]),atol=1e-6)
  for a,b in ((True,False),(False,True),(True,True)):np.testing.assert_allclose(contact_features(reflect_sequence(s,a,b)),x,atol=1e-6)
  order=[2,0,1,5,3,4];p={**s,'xy':s['xy'][:,order],'speed':s['speed'][:,order],'teams':s['teams'][order]};np.testing.assert_allclose(contact_features(p),x,atol=1e-6)
  full=contact_state(s,np.array([105,68]));np.testing.assert_allclose(swap_contact(swap_contact(full)),full,atol=0)
 def test_stationary_ball_has_no_turn_or_impulse_evidence(self):
  s=self.sequence();s['ball'][:]=0;x=contact_features(s)
  for offset in (0,32):self.assertTrue((x[:,offset+3:offset+5]==0).all())

class ContactTrainingTests(unittest.TestCase):
 def test_warm_start_preserves_old_network_and_window_targets(self):
  import torch
  from pathlib import Path
  from controlswitch.gated_rnn import load_checkpoint
  from controlswitch.contact_training import warm_contact,contact_window
  from controlswitch.spatial_symmetry import augmented_window
  old=load_checkpoint(Path(__file__).resolve().parents[1]/'models/archive/gru-short053-validation-only.npz');m,mean,scale=warm_contact(old,np.zeros(64,np.float32),np.ones(64,np.float32));m.eval();torch.set_num_threads(1);torch.manual_seed(57);x=torch.randn(2,24,572)
  with torch.inference_mode():
   a,_=old['model'](x[:,:,:508]);b,_=m(x)
  np.testing.assert_allclose(a.numpy(),b.numpy(),atol=1e-5,rtol=1e-5)
  np.testing.assert_array_equal(m.embedding[0].weight.detach().numpy()[:,508:],0)
  rng=np.random.default_rng(57);n=44;e={'x':rng.normal(size=(n,508)).astype(np.float32),'extra':rng.normal(size=(n,64)).astype(np.float32),'y':np.r_[np.zeros(20,int),np.full(4,-1),np.ones(20,int)],'focus_w':np.ones(n,np.float32)*2,'t':np.zeros(n,np.float32),'v':np.ones(n,np.float32)}
  for code in range(8):
   oldw=augmented_window(e['x'],e['y'],e['focus_w'],e['t'],e['v'],0,old['mean'],old['scale'],code=code,spatial=True);neww=contact_window(e,0,mean,scale,code);np.testing.assert_array_equal(oldw[0],neww[0][:,:508])
   for a,b in zip(oldw[1:],neww[1:]):np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
