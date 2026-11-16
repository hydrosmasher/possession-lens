import unittest,numpy as np
from controlswitch.gap_bridge import bridge_features,swap_bridge,anchored_gap_decode
class GapBridgeTests(unittest.TestCase):
    def scene(self):
        n=100;t=np.arange(n)/25;xy=np.tile([[-40.,0.],[-10.,2.],[10.,-2.],[40.,0.]],(n,1,1));xy[:,:,0]+=t[:,None];return {'xy':xy,'teams':np.array([0,0,1,1]),'ball':np.column_stack([t,np.zeros(n)]),'speed':np.ones((n,4)),'ball_speed':np.ones(n),'y':np.zeros(n,int)}
    def test_no_ball_invention_and_labels_ignored(self):
        s=self.scene();original=s['ball'].copy();mask=np.zeros(100,bool);mask[30:60]=True;x,active,seen=bridge_features(s,[105,68],mask);assert active.sum()==30;assert not seen[mask].any();np.testing.assert_array_equal(s['ball'],original);s['y'][:]=2;z,_,_=bridge_features(s,[105,68],mask);np.testing.assert_array_equal(x,z)
        mask[:10]=True;mask[-10:]=True;_,active,_=bridge_features(s,[105,68],mask);assert not active[:10].any() and not active[-10:].any()
    def test_team_swap_equivariance(self):
        s=self.scene();mask=np.zeros(100,bool);mask[30:60]=True;x,a,_=bridge_features(s,[105,68],mask);s['teams']=1-s['teams'];z,b,_=bridge_features(s,[105,68],mask);np.testing.assert_allclose(swap_bridge(x),z,atol=1e-6);np.testing.assert_array_equal(a,b)
    def test_decoder_locks_visible_and_dead_and_preserves_zero_mix(self):
        n=200;frozen=np.zeros(n,int);frozen[100]=2;prior=np.tile([.8,.2],(n,1));expert=np.tile([.001,.999],(n,1));active=np.zeros(n,bool);active[30:170]=True;p=anchored_gap_decode(frozen,prior,expert,active);np.testing.assert_array_equal(p[~active],frozen[~active]);np.testing.assert_array_equal(p==2,frozen==2);assert (p[35:45]==1).any();np.testing.assert_array_equal(anchored_gap_decode(frozen,prior,expert,active,mix=0),frozen)
        swap=np.where(frozen==2,2,1-frozen);z=anchored_gap_decode(swap,prior[:,::-1],expert[:,::-1],active);np.testing.assert_array_equal(z,np.where(p==2,2,1-p))

    def test_gaps_beyond_limit_are_not_bridged(self):
        original=self.scene();ix=np.arange(500)%100
        s={k:(v[ix] if isinstance(v,np.ndarray) and k!='teams' else v) for k,v in original.items()}
        mask=np.zeros(500,bool);mask[30:431]=True
        _,active,_=bridge_features(s,[105,68],mask)
        assert not active.any()
