import unittest
import numpy as np
from controlswitch.receipt_evidence import receipt_features, suppress_unsupported_pulses

class ReceiptEvidenceTests(unittest.TestCase):
    def scene(self, flyby=False):
        n=101;t=(np.arange(n)-50)/25
        ball=np.column_stack([t*12 if flyby else np.minimum(t,0)*12,np.zeros(n)])
        xy=np.tile([[0.,0.],[20.,4.]],(n,1,1))
        return {'xy':xy,'ball':ball,'teams':np.array([0,1]),'y':np.zeros(n,int)}
    def test_receipt_and_flyby_have_different_departure_evidence(self):
        receipt,rv=receipt_features(self.scene());flyby,fv=receipt_features(self.scene(True))
        assert rv[50,0] and fv[50,0]
        self.assertAlmostEqual(receipt[50,0,0],flyby[50,0,0])
        self.assertLess(receipt[50,0,8],flyby[50,0,8])
        self.assertGreater(receipt[50,0,11],flyby[50,0,11])
        self.assertGreater(receipt[50,0,13],flyby[50,0,13])
    def test_missing_ball_no_extrapolation_and_labels_ignored(self):
        s=self.scene();a,v=receipt_features(s);s['y'][:]=2;b,w=receipt_features(s)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(v,w)
        assert not v[:12].any() and not v[-12:].any()
        mask=np.zeros(101,bool);mask[51]=True;x,valid=receipt_features(s,mask)
        assert not valid[50].any();assert (x[50]==0).all()
    def test_reflection_translation_and_team_swap(self):
        s=self.scene(True);a,v=receipt_features(s)
        s['xy']=s['xy']*[-1,1]+[30,8];s['ball']=s['ball']*[-1,1]+[30,8]
        b,w=receipt_features(s);np.testing.assert_allclose(a,b,atol=1e-5);np.testing.assert_array_equal(v,w)
        s['teams']=1-s['teams'];c,z=receipt_features(s);np.testing.assert_allclose(c,b[:,::-1]);np.testing.assert_array_equal(z,w[:,::-1])
    def test_filter_preserves_supported_short_receipt_and_missing_evidence(self):
        y=np.r_[np.zeros(20,int),np.ones(5,int),np.zeros(20,int),[2]]
        p=np.tile([.9,.1],(len(y),1));v=np.ones_like(p,bool)
        fixed,edits=suppress_unsupported_pulses(y,p,v,.5);assert len(edits)==1;assert (fixed[20:25]==0).all();assert fixed[-1]==2
        p[22,1]=.8;fixed,_=suppress_unsupported_pulses(y,p,v,.5);np.testing.assert_array_equal(fixed,y)
        p[22,1]=.1;v[22]=False;fixed,_=suppress_unsupported_pulses(y,p,v,.5);np.testing.assert_array_equal(fixed,y)
