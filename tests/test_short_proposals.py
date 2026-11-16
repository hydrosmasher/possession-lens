import unittest,numpy as np
from controlswitch.short_proposals import recover_short
class ShortProposalTests(unittest.TestCase):
    def inputs(self):
        p=np.zeros(200,int);proposed=p.copy();proposed[50:65]=1;seen=np.ones(200,bool);score=np.tile([.9,.1],(200,1));valid=np.ones((200,2),bool);close=valid.copy();return p,proposed,seen,score,valid,close
    def test_short_real_support_recovers_without_duration_floor(self):
        a=self.inputs();a[3][56,1]=.9;p,records=recover_short(*a,.5);assert (p[50:65]==1).all();assert len(records)==1 and records[0]['accepted'];np.testing.assert_array_equal(p[:50],a[0][:50]);np.testing.assert_array_equal(p[65:],a[0][65:])
    def test_no_support_missing_ball_and_dead_are_preserved(self):
        a=self.inputs();p,_=recover_short(*a,.5);np.testing.assert_array_equal(p,a[0]);a[3][56,1]=.9;a[2][55]=False;p,_=recover_short(*a,.5);np.testing.assert_array_equal(p,a[0]);a[2][55]=True;a[0][55]=2;p,_=recover_short(*a,.5);np.testing.assert_array_equal(p,a[0])
    def test_far_or_unavailable_receipt_is_not_support(self):
        a=self.inputs();a[3][56,1]=.9;a[5][56,1]=False;p,_=recover_short(*a,.5);np.testing.assert_array_equal(p,a[0]);a[5][56,1]=True;a[4][56,1]=False;p,_=recover_short(*a,.5);np.testing.assert_array_equal(p,a[0])
    def test_team_swap_and_long_possessions(self):
        a=self.inputs();a[3][56,1]=.9;p,_=recover_short(*a,.5);z,_=recover_short(1-a[0],1-a[1],a[2],a[3][:,::-1],a[4][:,::-1],a[5][:,::-1],.5);np.testing.assert_array_equal(z,1-p);a[1][50:150]=1;p,_=recover_short(*a,.5);np.testing.assert_array_equal(p,a[0])
