import unittest,tempfile
from pathlib import Path
import numpy as np
from controlswitch.proposal_filter import save,load,filter_proposals
class PortableProposalFilterTests(unittest.TestCase):
    def model(self):return {'weights':np.zeros((12,2),np.float32),'mean':np.zeros(11,np.float32),'scale':np.ones(11,np.float32)}
    def test_rejected_artifact_is_not_accidentally_deployed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'filter.npz';save(p,self.model(),.4,{'external_only':True})
            with self.assertRaises(ValueError):load(p)
            b=load(p,allow_research=True);current=np.zeros(100,int);events=[{'start':40,'end':50,'team':1}];out,scores=filter_proposals(b,current,events,np.zeros((1,17)));assert (out[40:50]==1).all();assert scores[0]==.5;np.testing.assert_array_equal(out[:40],current[:40])
    def test_rejects_bad_scaling(self):
        m=self.model();m['scale'][0]=0
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):save(Path(tmp)/'bad.npz',m,.3,{})

    def test_filter_cannot_overwrite_dead_ball_labels(self):
        from controlswitch.proposal_calibration import apply_scores
        current=np.zeros(100,int);current[45]=2
        with self.assertRaises(ValueError):apply_scores(current,[{'start':40,'end':50,'team':1}],[.9],.5)
