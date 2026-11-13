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

 def test_decoder_study_uses_full_prediction_counts_not_censored_fragments(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'reports/external';p.mkdir(parents=True)
   diag={c:{'predicted_spells_under_0_4_seconds':4,'full_prediction_spells_under_0_4_seconds':0} for c in ['clean','masked']}
   report={'reference_validation':{'selection_spell_f1':.5527},'reference_diagnostics':diag,'selected':{'selection_spell_f1':.5554,'diagnostics':diag}}
   (p/'rnn-gated-decoder-004-report.json').write_text(json.dumps(report));r=research_summary(root);self.assertEqual(r['candidate'],.5554);self.assertEqual(r['short_pulses_candidate'],0);self.assertEqual(r['short_pulses_reference'],0)

 def test_rejected_followup_does_not_replace_saved_candidate(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'reports/external';p.mkdir(parents=True)
   (p/'rnn-gated-decoder-004-report.json').write_text(json.dumps({'reference_validation':{'selection_spell_f1':.55},'selected':{'selection_spell_f1':.5554}}))
   (p/'rnn-gated-recovery-006-report.json').write_text(json.dumps({'selected':None,'best_unrestricted':{'selection_spell_f1':.5791}}))
   r=research_summary(root);self.assertEqual(r['candidate'],.5554);self.assertFalse(r['latest_attempt']['accepted']);self.assertEqual(r['latest_attempt']['score'],.5791)
