import json,tempfile,unittest
from pathlib import Path
import numpy as np
from controlswitch.ranked_pipeline import make_bundle,infer,load_pipeline,REFERENCE,VERSION
from controlswitch.control_correction import load_candidate,infer_candidate

ROOT=Path(__file__).resolve().parents[1]

class RankedPipelineTests(unittest.TestCase):
    def test_reference_matches_existing019_probabilities_and_labels_exactly(self):
        paths={'team':ROOT/'models/rnn-europe-context-001-experimental.npz','control':ROOT/'models/archive/control-017-validation-only.npz','dead_blend':ROOT/'models/archive/deadblend-019-validation-only.npz'}
        new=make_bundle(REFERENCE,paths);old=load_candidate(ROOT/'reports/external/deadblend-019-settings.json',paths['team'],paths['dead_blend'],paths['control'])
        rng=np.random.default_rng(30);n=80;s={'xy':rng.normal(size=(n,4,2)),'speed':np.ones((n,4)),'teams':np.array([0,0,1,1]),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)}
        mask=np.arange(n)%3==0
        a,p=infer(new,s,np.array([105,68]),mask);b,q=infer_candidate(old,s,np.array([105,68]),mask)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(p,q)
    def test_unknown_specification_and_escaping_asset_are_rejected(self):
        with self.assertRaises(ValueError):make_bundle({'name':'unknown'}, {})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';assets={k:{'path':'../escape.npz','sha256':'invalid'} for k in ('team','control','dead_blend')};p.write_text(json.dumps({'version':VERSION,'spec':REFERENCE,'assets':assets}))
            with self.assertRaises(ValueError):load_pipeline(p,Path(tmp))
if __name__=='__main__':unittest.main()
