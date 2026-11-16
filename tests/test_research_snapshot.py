import unittest,json,numpy as np
from pathlib import Path
from controlswitch.research_snapshot import make_snapshot,infer_snapshot,cached_case
from controlswitch.recurrent_pipeline import load_pipeline,infer
class SnapshotTests(unittest.TestCase):
 def test_frozen053_matches_runtime_and_cache(self):
  root=Path(__file__).resolve().parents[1];roots=json.loads((root/'local/roots.json').read_text()) if (root/'local/roots.json').exists() else None
  if roots is None:self.skipTest('Private research lineage absent from clean source')
  descriptor={'source':'short-spell-053','name':'short_focus','control_source':'short-spell-053','control_name':'short_focus'};b=make_snapshot(descriptor,0,roots,root/'local');runtime=load_pipeline(root/'reports/external/short-spell-053-config.json');rng=np.random.default_rng(60);n=75;s={'xy':rng.normal(size=(n,4,2)),'teams':np.array([0,0,1,1]),'speed':np.ones((n,4)),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)};pitch=np.array([105,68]);p,q=infer_snapshot(b,s,pitch);a,z=infer(runtime,s,pitch);np.testing.assert_array_equal(p,a);np.testing.assert_array_equal(q,z);y,p,q=cached_case(descriptor,0,0,root/'local');self.assertEqual(q.shape,(len(y),3));self.assertEqual(p.shape,y.shape)
if __name__=='__main__':unittest.main()
