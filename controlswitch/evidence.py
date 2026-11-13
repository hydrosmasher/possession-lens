"""Shareable external metrics only; no challenge-derived analysis is read."""
from pathlib import Path
import json,html,math

def summary(report):
 if 'test_results' in report:
  results=report['test_results'];mapping={'baseline':'baseline','current':'previous_european','candidate':'candidate'};values={}
  for key,source in mapping.items():
   row=results[source];a=float(row['clean']['spell_f1_mean']);b=float(row['masked']['spell_f1_mean']);mean=float(row['selection_spell_f1'])
   if not all(math.isfinite(v) and 0<=v<=1 for v in (a,b,mean)) or not math.isclose(mean,(a+b)/2,abs_tol=1e-10):raise ValueError('External evidence metrics do not reconcile')
   values[key]={'clean':a,'dropout':b,'mean':mean}
  delta=values['candidate']['mean']-values['current']['mean']
  return {'match_id':'2841','match_title':'Barcelona–Real Madrid · 18 December 2019','models':values,'labels':{'baseline':'Baseline','current':'Previous European RNN','candidate':'Context RNN'},'candidate_minus_current':delta,'candidate_wins':delta>0,'note':'Frozen fresh-match comparison. Partial automatic H/A labels, not full-match challenge accuracy. No dead-ball evaluation. Target 0.8 remains unmet.','source':'SkillCorner historical open data, MIT.'}
 results=report['fresh_match_results'];values={}
 for key in ('baseline','current','candidate'):
  row=results[key];clean=float(row['clean']['spell_f1_mean']);masked=float(row['masked']['spell_f1_mean']);mean=float(row['mean_clean_masked_spell_f1'])
  if not all(math.isfinite(v) and 0<=v<=1 for v in (clean,masked,mean)) or not math.isclose(mean,(clean+masked)/2,abs_tol=1e-10):raise ValueError('External evidence metrics do not reconcile')
  values[key]={'clean':clean,'dropout':masked,'mean':mean}
 delta=values['candidate']['mean']-values['current']['mean']
 return {'match_id':'J03WR9','match_title':report['source']['metadata']['title'],'models':values,'candidate_minus_current':delta,'candidate_wins':delta>0,'note':'Previously unused match, frozen models. Shares a club with training. F1 is a possession-segmentation metric, not match accuracy. One match is limited evidence. No challenge truth or video checks.','source':'Bassek et al. (2025), IDSSE/Sportec, doi:10.1038/s41597-025-04505-y; CC BY 4.0.'}

def export_report(report_path,out):
 report=json.loads(Path(report_path).read_text());s=summary(report);parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Possession Lens · Fresh match test</title><style>body{margin:0;background:#080f1d;color:#e9f0ff;font:16px/1.7 system-ui}main{max-width:1000px;margin:50px auto;padding:24px}h1{font-size:40px;line-height:1.15;color:#f4c849}h2{color:#90bfff}table{width:100%;border-collapse:collapse;background:#112039;border-radius:10px}th,td{text-align:left;padding:14px;border-bottom:1px solid #ffffff15}p{color:#b5c3dc}.verdict{padding:20px;border-left:4px solid #f4c849;background:#122039}a{color:#8cbdff}small{color:#9faecc}</style><main><p>POSSESSION LENS / FROZEN EXTERNAL TEST</p><h1>'+html.escape(s['match_title'])+'</h1><p class="verdict">'+('The improved RNN beat the current model on the declared primary metric.' if s['candidate_wins'] else 'The improved RNN did not beat the current model on the declared primary metric.')+'</p><p>Mean clean/dropout spell-F1 difference: '+f"{s['candidate_minus_current']:+.4f}"+'. No parameters were changed after viewing this match.</p><table><tr><th>Model</th><th>Clean spell F1</th><th>Ball dropout spell F1</th><th>Mean (primary)</th></tr>']
 for key,label in [('baseline','Reference baseline'),('current','Current boundary RNN'),('candidate','Longer-trained candidate')]:
  m=s['models'][key];parts.append(f'<tr><td>{label}</td><td>{m["clean"]:.4f}</td><td>{m["dropout"]:.4f}</td><td>{m["mean"]:.4f}</td></tr>')
 parts.append('</table><h2>Switching and missed possessions</h2><table><tr><th>Condition / model</th><th>Missed spells</th><th>False positives</th><th>Under 0.4 s</th><th>Merge proxy</th></tr>')
 for condition in ('clean','masked'):
  for key in ('current','candidate'):
   d=report['diagnostics'][key][condition];parts.append(f'<tr><td>{condition} / {key}</td>'+''.join(f'<td>{d[f]}</td>' for f in ('missed_reference_spells_at_iou_0_5','false_positive_spells_at_iou_0_5','predicted_spells_under_0_4_seconds','predictions_overlapping_multiple_reference_spells'))+'</tr>')
 parts.append('</table><p>Short-spell and merge counts are descriptive proxies. They do not establish that every short prediction is wrong.</p><h2>Why this is a fresh check</h2><p>The candidate and reference checkpoints, seed 300, baseline window 50 and primary metric were recorded before downloading or scoring J03WR9. Candidate: rnn-capacity-002-1layer-experimental.npz. Current model: rnn-boundary-001.npz. Training used J03WOH; validation used J03WPY. No fitting or tuning occurred on this match.</p><h2>What remains uncertain</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in report['limitations'])+'</ul><p>Models remain frozen. The default is not automatically replaced by this report.</p><small>Data: <a href="https://doi.org/10.1038/s41597-025-04505-y">Bassek et al. (2025), IDSSE/Sportec</a>, CC BY 4.0. Full source and model SHA-256 values, all metrics and the preregistered scoring plan are in the adjacent JSON.</small></main>')
 Path(out).write_text('\n'.join(parts));return s


def research_summary(root):
    """Latest eligible validation result; never replaces frozen fresh-test evidence."""
    from pathlib import Path
    import json
    attempt=None
    for filename,label in [('rnn-event-duration-012-report.json','short-spell duration check'),('rnn-event-precision-011-report.json','compact short-spell verifier'),('rnn-event-precision-010-report.json','paired-switch verifier'),('rnn-selective-precision-009-report.json','selective verifier'),('rnn-crossfit-precision-008-report.json','match-excluded verifier'),('rnn-boundary-precision-007-report.json','boundary verifier'),('rnn-gated-recovery-006-report.json','fine-tuning')]:
        attempt_path=Path(root)/'reports/external'/filename
        if attempt_path.exists():
            study=json.loads(attempt_path.read_text())
            attempt={'score':study['best_unrestricted']['selection_spell_f1'],'accepted':study.get('selected') is not None,'label':label}
            break
    for filename,label in [('rnn-gated-decoder-004-report.json','Refined GRU boundary decoder'),('rnn-gated-003-report.json','Gated RNN'),('rnn-interaction-context-001-report.json','Interaction RNN')]:
        path=Path(root)/'reports/external'/filename
        if not path.exists():continue
        report=json.loads(path.read_text());candidate=report.get('selected')
        if candidate:
            return {'name':({'gru1':'GRU','bigru2':'Bidirectional GRU','bilstm2':'Bidirectional LSTM','bigru2_boundary':'Bidirectional GRU + boundary head'}.get(candidate.get('variant'),label)),
                    'reference':report['reference_validation']['selection_spell_f1'],
                    'candidate':candidate['selection_spell_f1'],'fresh_test':False,'promoted':False,'latest_attempt':attempt,
                    'short_pulses_reference':sum(report['reference_diagnostics'][c]['full_prediction_spells_under_0_4_seconds'] for c in ('clean','masked')) if report.get('reference_diagnostics') and all('full_prediction_spells_under_0_4_seconds' in report['reference_diagnostics'][c] for c in ('clean','masked')) else None,
                    'short_pulses_candidate':sum(candidate['diagnostics'][c]['full_prediction_spells_under_0_4_seconds'] for c in ('clean','masked')) if candidate.get('diagnostics') and all('full_prediction_spells_under_0_4_seconds' in candidate['diagnostics'][c] for c in ('clean','masked')) else None}
    return None
