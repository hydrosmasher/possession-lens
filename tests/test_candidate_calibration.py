import unittest,numpy as np
from controlswitch.candidate_calibration import positive_retention_threshold,pool_targets,candidate_pool,COLUMNS
class CandidateCalibrationTests(unittest.TestCase):
    def test_cutoff_retains_every_positive_and_abstains_without_positives(self):
        p=np.array([.8,.2,.5,.1]);y=np.array([1,0,1,0]);t=positive_retention_threshold(p,y);assert (p[y==1]>=t).all();self.assertEqual(t,.5);self.assertEqual(positive_retention_threshold(p,np.zeros(4,int)),0)
    def test_proxy_targets_use_iou_and_exclude_unknown(self):
        y=np.r_[np.zeros(80,int),np.ones(20,int),np.zeros(100,int)];events=[{'start':80,'end':100,'team':1},{'start':140,'end':145,'team':1}];labels,known=pool_targets(y,events);np.testing.assert_array_equal(labels,[1,0]);assert known.all();y[85]=3;_,known=pool_targets(y,events);assert not known[0]
    def test_scoring_columns_do_not_use_baseline_context_shortcut(self):
        assert set(COLUMNS).isdisjoint({1,2,3,4,15,16})

    def test_pool_features_ignore_baseline_correctness_and_team_names(self):
        n=100;proposal=np.ones(n,int);proposal[40:50]=0;g=np.tile([.2,.8],(n,1));g[40:50]=[.8,.2]
        case={'proposal':proposal,'seen':np.ones(n,bool),'current':np.ones(n,int),'q':np.tile([.1,.8,.1],(n,1)),'g':g,'motion':np.ones((n,2,14)),'receipt_valid':np.ones((n,2),bool),'receipt':np.tile([.9,.1],(n,1))}
        events,a=candidate_pool(case,.5);assert len(events)==1
        case['current']=np.zeros(n,int);case['q']=np.tile([.8,.1,.1],(n,1));_,b=candidate_pool(case,.5);np.testing.assert_array_equal(a,b)
        swapped={**case,'proposal':1-case['proposal'],'current':1-case['current'],'q':case['q'][:,[1,0,2]],'g':g[:,::-1],'motion':case['motion'][:,::-1],'receipt_valid':case['receipt_valid'][:,::-1],'receipt':case['receipt'][:,::-1]}
        _,c=candidate_pool(swapped,.5);np.testing.assert_array_equal(a,c)
