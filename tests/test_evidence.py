import unittest
from controlswitch.evidence import summary
class EvidenceTests(unittest.TestCase):
 def test_summary_uses_declared_mean_and_rejects_inconsistent_scores(self):
  row=lambda a,b:{'clean':{'spell_f1_mean':a},'masked':{'spell_f1_mean':b},'mean_clean_masked_spell_f1':(a+b)/2}
  r={'source':{'metadata':{'title':'External example'}},'fresh_match_results':{'baseline':row(.2,.1),'current':row(.4,.3),'candidate':row(.5,.4)}}
  s=summary(r);self.assertTrue(s['candidate_wins']);self.assertAlmostEqual(s['candidate_minus_current'],.1)
  r['fresh_match_results']['candidate']['mean_clean_masked_spell_f1']=.9
  with self.assertRaises(ValueError):summary(r)
