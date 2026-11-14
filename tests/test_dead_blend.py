import unittest,tempfile
from pathlib import Path
import numpy as np
from controlswitch.dead_blend import combine,save_combined
from controlswitch import tree_model

def stump(baseline,values):
    return {'meta':{'version':'possession-trees-v1','kind':'hist_boost','count':1,'features':2,'classes':[0,1],'baseline':baseline},'trees':[{'left':np.array([1,-1,-1]),'right':np.array([2,-1,-1]),'feature':np.array([0,0,0]),'threshold':np.array([0.,0,0]),'value':np.array([0.,*values]),'missing':np.array([1,0,0])}]}

class DeadBlendTests(unittest.TestCase):
    def test_matches_weighted_log_odds_and_does_not_mutate_sources(self):
        a=stump(.2,[-.3,.8]);b=stump(-.1,[.7,-.2]);x=np.array([[-1,0],[1,0],[np.nan,0]])
        before=a['trees'][0]['value'].copy();mix=.25;p=tree_model.predict(combine(a,b,mix),x)[:,1]
        pa=tree_model.predict(a,x)[:,1];pb=tree_model.predict(b,x)[:,1]
        logits=(1-mix)*np.log(pa/(1-pa))+mix*np.log(pb/(1-pb));np.testing.assert_allclose(p,1/(1+np.exp(-logits)))
        np.testing.assert_array_equal(a['trees'][0]['value'],before)
    def test_endpoints_and_pickle_free_roundtrip(self):
        a=stump(.2,[-.3,.8]);b=stump(-.1,[.7,-.2]);x=np.array([[-1,0],[1,0]])
        for mix,expected in [(0,a),(1,b)]:np.testing.assert_allclose(tree_model.predict(combine(a,b,mix),x),tree_model.predict(expected,x))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'blend.npz';model=combine(a,b,.5);save_combined(p,model);np.testing.assert_allclose(tree_model.predict(tree_model.load(p),x),tree_model.predict(model,x))
    def test_invalid_blend_rejected(self):
        for weight in (np.nan,-.1,1.1):
            with self.assertRaises(ValueError):combine(stump(.1,[.1,.2]),stump(.1,[.1,.2]),weight)
if __name__=='__main__':unittest.main()
