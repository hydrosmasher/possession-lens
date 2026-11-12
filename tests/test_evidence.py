import unittest
from controlswitch.evidence import summary
class EvidenceTests(unittest.TestCase):
 def test_summary_uses_declared_mean_and_rejects_inconsistent_scores(self):
  row=lambda a,b:{'clean':{'spell_f1_mean':a},'masked':{'spell_f1_mean':b},'mean_clean_masked_spell_f1':(a+b)/2}
  r={'source':{'metadata':{'title':'External example'}},'fresh_match_results':{'baseline':row(.2,.1),'current':row(.4,.3),'candidate':row(.5,.4)}}
  s=summary(r);self.assertTrue(s['candidate_wins']);self.assertAlmostEqual(s['candidate_minus_current'],.1)
  r['fresh_match_results']['candidate']['mean_clean_masked_spell_f1']=.9
  with self.assertRaises(ValueError):summary(r)
 def test_context_evidence_uses_same_test_european_comparator(self):
  import json
  from pathlib import Path
  report=json.loads((Path(__file__).resolve().parents[1]/'reports/external/rnn-europe-context-001-report.json').read_text());s=summary(report)
  self.assertEqual(s['match_id'],'2841');self.assertEqual(s['labels']['current'],'Previous European RNN');self.assertAlmostEqual(s['models']['current']['mean'],.5803569925065253);self.assertTrue(s['candidate_wins'])
