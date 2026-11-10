import unittest
import numpy as np
from controlswitch.model import features,decode,boundary_targets,fit_classifier,probabilities,causal_features,turnover_targets,interaction_features
from controlswitch.experiment import aggregate,crop_sequence

class ModelTests(unittest.TestCase):
 def test_decoder_matches_exhaustive_optimum(self):
    from itertools import product
    p=np.array([[.7,.2,.1],[.2,.6,.2],[.1,.7,.2]])
    actual=decode(p,.8)
    def score(seq):return sum(np.log(p[i,k]) for i,k in enumerate(seq))-.8*sum(seq[i]!=seq[i-1] for i in range(1,3))
    best=max(product(range(3),repeat=3),key=score)
    self.assertEqual(tuple(actual),best)
 def test_boundary_evidence_can_preserve_brief_genuine_possession(self):
    p=np.tile([.9,.09,.01],(15,1));p[6:9]=[.08,.91,.01]
    fixed=decode(p,5)
    b=np.zeros(15);b[[6,9]]=1
    verified=decode(p,5,b,1)
    self.assertTrue(np.all(fixed==0));self.assertTrue(np.all(verified[6:9]==1))
 def test_reference_labels_cannot_affect_features(self):
    n=100;s={'xy':np.ones((n,2,2)),'speed':np.ones((n,2)),'teams':np.array([0,1]),
             'ball':np.zeros((n,2)),'ball_speed':np.ones(n),'y':np.zeros(n,dtype=int)}
    x=features(s,[105,68]);s['y'][:]=2
    np.testing.assert_array_equal(x,features(s,[105,68]))
    self.assertTrue(np.isfinite(features(s,[105,68],ball_missing=np.ones(n,bool))).all())
 def test_classifier_learns_three_states(self):
    rng=np.random.default_rng(42);x=np.concatenate([rng.normal(k*4,.2,(100,2)) for k in (-1,0,1)]);y=np.repeat([0,1,2],100)
    m=fit_classifier(x,y,epochs=50)
    self.assertGreater((probabilities(m,x).argmax(axis=1)==y).mean(),.95)
 def test_period_separator_prevents_merging(self):
    y=np.zeros(50,dtype=np.int8);r=aggregate([(y,y),(y,y)])
    self.assertEqual(r['n_gt_spells'],2);self.assertEqual(r['spell_f1_mean'],1)
 def test_unknown_label_is_not_a_boundary_target(self):
    y=np.array([0,3,1],dtype=np.int8)
    self.assertFalse(boundary_targets(y).any())
 def test_forecast_features_cannot_see_future(self):
    n=100;s={'xy':np.ones((n,2,2)),'speed':np.ones((n,2)),'teams':np.array([0,1]),
             'ball':np.zeros((n,2)),'ball_speed':np.ones(n)}
    original=causal_features(s,[105,68])
    s['xy'][50:]=999;s['speed'][50:]=99;s['ball'][50:]=999;s['ball_speed'][50:]=99
    np.testing.assert_array_equal(original[:50],causal_features(s,[105,68])[:50])
 def test_forecast_target_excludes_restarts_and_incomplete_horizons(self):
    y=np.array([0,0,2,1,1,0,0,0],dtype=np.int8)
    target=turnover_targets(y,3)
    self.assertEqual(target[0],0);self.assertEqual(target[2],-1)
    self.assertEqual(target[3],1);self.assertEqual(target[-1],-1)
 def test_interaction_evidence_depends_on_proximity_and_movement(self):
    ball=np.array([[0,0],[.1,0],[.1,.1],[.1,.2]],dtype=float)
    xy=np.zeros((4,2,2));xy[:,0]=ball;xy[:,1]=50
    s={'xy':xy,'teams':np.array([0,1]),'ball':ball}
    evidence=interaction_features(s)
    self.assertGreater(evidence[2,1],evidence[2,6]);self.assertGreater(evidence[2,0],0)

if __name__=='__main__':unittest.main()
