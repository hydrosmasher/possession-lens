import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
from controlswitch.rnn import initialise,normalise,loss_gradients
from controlswitch.model import decode
from controlswitch.rnn_refine import transition_weights,read_cache

class RefinementTests(unittest.TestCase):
 def test_weighted_bptt_finite_differences_with_unknown_label(self):
    x=np.array([[.2,.4],[.4,.1],[.3,-.2]],dtype=np.float32);y=np.array([0,3,2]);w=np.array([1,6,3.],dtype=np.float32)
    m=initialise(x,hidden=3);z=normalise(m,x);_,g,_=loss_gradients(m,z,y,weights=w)
    for k,ix in [('wx',(0,1)),('wh',(1,2)),('wy',(1,0)),('bh',(0,)),('by',(2,))]:
        original=m[k][ix];eps=.002
        m[k][ix]=original+eps;plus=loss_gradients(m,z,y,weights=w)[0]
        m[k][ix]=original-eps;minus=loss_gradients(m,z,y,weights=w)[0];m[k][ix]=original
        self.assertAlmostEqual(float(g[k][ix]),(plus-minus)/(2*eps),delta=.0002)
 def test_weights_do_not_invent_transitions_across_unknown_labels(self):
    y=np.array([0,0,3,1,1,1,0,0]);original=y.copy()
    np.testing.assert_array_equal(transition_weights(y,boost=6,radius=0),[1,1,0,1,1,1,6,1])
    np.testing.assert_array_equal(y,original)
 def test_decoder_matches_exhaustive_short_path_search(self):
    p=np.array([[.7,.2,.1],[.1,.4,.5],[.1,.8,.1],[.2,.2,.6]])
    team=.5;stop=.1
    def score(path):
        return sum(np.log(p[i,s]) for i,s in enumerate(path))-sum((stop if 2 in (a,b) else team) for a,b in zip(path,path[1:]) if a!=b)
    expected=max(itertools.product(range(3),repeat=len(p)),key=score)
    np.testing.assert_array_equal(decode(p,team,stop_penalty=stop),expected)
 def test_equal_penalties_preserve_legacy_decoder(self):
    p=np.random.default_rng(5).dirichlet([1,1,1],100)
    np.testing.assert_array_equal(decode(p,6),decode(p,6,stop_penalty=6))
 def test_viewed_test_cannot_be_used_as_training_or_validation(self):
    with self.assertRaises(ValueError):read_cache('file-not-opened.npz','J03WMX')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'bad.npz';np.savez(path,match_id=np.array('J03WMX'),source_sha256=np.array('wrong'))
        with self.assertRaises(ValueError):read_cache(path,'J03WPY')
