import unittest,tempfile,json
from pathlib import Path
from controlswitch.evidence import research_summary
class ResearchSummaryTests(unittest.TestCase):
 def test_latest_eligible_only_and_always_validation_only(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'reports/external';p.mkdir(parents=True)
   self.assertIsNone(research_summary(root))
   report={'reference_validation':{'selection_spell_f1':.5},'selected':{'selection_spell_f1':.51}}
   (p/'rnn-interaction-context-001-report.json').write_text(json.dumps(report));self.assertEqual(research_summary(root)['candidate'],.51)
   (p/'rnn-gated-003-report.json').write_text(json.dumps({'selected':None}));self.assertEqual(research_summary(root)['candidate'],.51)
   report['selected']={'selection_spell_f1':.54,'variant':'gru1'};(p/'rnn-gated-003-report.json').write_text(json.dumps(report));value=research_summary(root);self.assertEqual(value['candidate'],.54);self.assertFalse(value['fresh_test']);self.assertFalse(value['promoted'])
