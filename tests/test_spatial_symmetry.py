import unittest,numpy as np
try:
 from controlswitch.spatial_symmetry import reflection_signs,reflect_sequence,augmented_window
 from controlswitch.joint_state import joint_features
except ImportError:
 reflection_signs=None

@unittest.skipIf(reflection_signs is None,'Optional PyTorch training helpers')
class SpatialReflectionTests(unittest.TestCase):
 def test_feature_reflections_match_raw_geometry_recomputation(self):
  rng=np.random.default_rng(49);n=120;s={'xy':np.cumsum(rng.normal(0,.1,(n,4,2)),axis=0),'speed':np.ones((n,4)),'teams':np.array([0,0,1,1]),'ball':np.cumsum(rng.normal(0,.1,(n,2)),axis=0),'ball_speed':np.ones(n)};s['xy'][:,:2,0]-=20;s['xy'][:,2:,0]+=20;mask=np.zeros(n,bool);mask[20:50]=True;pitch=np.array([105,68]);x=joint_features(s,pitch,mask)
  for fx,fy in ((True,False),(False,True),(True,True)):
   y=joint_features(reflect_sequence(s,fx,fy),pitch,mask);np.testing.assert_allclose(x*reflection_signs(fx,fy),y,atol=1e-6)
 def test_window_normalisation_preserves_padding_and_swaps_only_known_labels(self):
  n=20;x=np.ones((n,508),np.float32);y=np.r_[np.zeros(10,int),np.full(10,-1)];w=np.ones(n,np.float32);t=np.zeros(n,np.float32);v=np.ones(n,np.float32);mean=np.ones(508,np.float32);scale=np.ones(508,np.float32);xx,yy,ww,tt,vv=augmented_window(x,y,w,t,v,0,mean,scale,code=1)
  self.assertTrue((xx[:64]==0).all());self.assertTrue((xx[84:]==0).all());self.assertTrue((yy[64:74]==1).all());self.assertTrue((ww[yy<0]==0).all())
if __name__=='__main__':unittest.main()
