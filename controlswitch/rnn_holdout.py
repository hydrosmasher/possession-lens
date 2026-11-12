"""One-shot frozen comparison on preregistered, previously unused J03WR9."""
from pathlib import Path
import argparse,json
from .pipeline import digest
from .rnn_fresh_test import verify_metadata
from .external import load_sportec,period_arrays
from .rnn_runtime import load_frozen,infer_frozen
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import spell_diagnostics
from .experiment import baseline_prediction

def verify_frozen(protocol,candidate,reference):
 if protocol.get('match_id')!='J03WR9' or protocol.get('tuning_on_this_match') is not False or protocol.get('competition_data_used') is not False:raise ValueError('Invalid untouched-match protocol')
 for name,path in [('candidate',candidate),('reference',reference)]:
  filename=protocol.get(name)
  if filename not in protocol.get('checkpoint_sha256',{}) or digest(path)!=protocol['checkpoint_sha256'][filename]:raise ValueError('Frozen checkpoint differs from preregistration')
 if protocol.get('baseline_window')!=50 or protocol.get('dropout_seed')!=300:raise ValueError('Do not change scoring settings after preregistration')

def run(tracking,metadata,candidate,reference,out):
 out=Path(out);protocol=json.loads((out/'preregistered-protocol.json').read_text());verify_frozen(protocol,candidate,reference)
 if (out/'report.json').exists() or (out/'evaluation-started.json').exists():raise ValueError('This fresh evaluation has already started. Preserve it; do not repeat or tune.')
 expected_source='70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269'
 if digest(tracking)!=expected_source:raise ValueError('Fresh tracking source differs from the published SHA-256; do not score it')
 identity=verify_metadata(metadata,'J03WR9');frozen={'candidate':load_frozen(candidate),'current':load_frozen(reference)}
 started={'protocol_sha256':digest(out/'preregistered-protocol.json'),'metadata':identity,'tracking_sha256':digest(tracking),'test_started':True,'training_on_test':False}
 (out/'evaluation-started.json').write_text(json.dumps(started,indent=2)+'\n')
 data=load_sportec(tracking,'J03WR9',out.parent/'external-cache');rows=conditions([(period_arrays(data,i),data['pitch']) for i in (0,1)],300)
 results={'baseline':evaluate(rows,[baseline_prediction(s,50,mask) for s,p,c,mask in rows])};diagnostics={}
 for name,state in frozen.items():
  labels=[infer_frozen(state,s,p,mask)[0] for s,p,c,mask in rows];results[name]=evaluate(rows,labels);diagnostics[name]=spell_diagnostics(rows,labels)
 for row in results.values():row['mean_clean_masked_spell_f1']=row.pop('selection_spell_f1')
 delta=results['candidate']['mean_clean_masked_spell_f1']-results['current']['mean_clean_masked_spell_f1']
 report={'protocol':protocol,'source':started,'fresh_match_results':results,'diagnostics':diagnostics,'candidate_minus_current_primary':delta,'candidate_beats_current_primary':delta>0,'candidate_beats_baseline_primary':results['candidate']['mean_clean_masked_spell_f1']>results['baseline']['mean_clean_masked_spell_f1'],'competition_accuracy_measured':False,'default_model_changed':False,'limitations':['One previously unused match; not proof of statistical significance or general superiority.','Fortuna Düsseldorf appears in training and validation, so clubs are not fully unseen.','This match is now viewed; no parameter tuning may be justified as fresh-test selection.','Simulated missing ball observations do not reproduce all competition tracking errors.','No prediction was compared against challenge video or challenge truth.']}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('tracking','metadata','candidate','reference','out'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();r=run(a.tracking,a.metadata,a.candidate,a.reference,a.out);print(json.dumps({k:r[k] for k in ['fresh_match_results','candidate_minus_current_primary','candidate_beats_current_primary','candidate_beats_baseline_primary']},indent=2))
if __name__=='__main__':main()
