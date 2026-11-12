"""Frozen external match evaluation, separate from all training and tuning commands."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
from .external import FRESH_MATCHES,HOLDOUT_MATCHES,load_sportec,period_arrays
from .pipeline import digest
from .model import instantaneous_features,decode,probabilities
from .rnn_boundaries import BOUNDARY_VERSION,invariant_boundary_features,dwell_decode,spell_diagnostics
from .rnn import observation_probabilities,RNN_VERSION,RNN_REFINED_VERSION,RNN_SYMMETRIC_VERSION,RNN_DWELL_VERSION
from .rnn_benchmark import conditions,evaluate
from .experiment import baseline_prediction

def verify_metadata(path,match='J03WN1'):
    if match not in {**FRESH_MATCHES,**HOLDOUT_MATCHES}:raise ValueError('Not a designated fresh external match')
    root=ET.parse(path).getroot();general=root.find('.//General')
    spec={**FRESH_MATCHES,**HOLDOUT_MATCHES}[match]
    if general is None or general.get('MatchId')!='DFL-MAT-'+match:raise ValueError('Metadata match identity mismatch')
    if general.get('HomeTeamId')!=spec['home'] or general.get('GuestTeamId')!=spec['away']:raise ValueError('Metadata home/away roles mismatch')
    if general.get('CompetitionId')!='DFL-COM-'+spec['competition']:raise ValueError('Metadata competition mismatch')
    return {'match_id':match,'title':general.get('MatchTitle'),'metadata_sha256':digest(path)}

def run(tracking,metadata,checkpoint,out,reference_checkpoint=None,match='J03WN1',secondary_checkpoint=None):
    from .rnn_runtime import load_frozen,infer_frozen
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output folder; repeated tests are not fresh evidence')
    identity=verify_metadata(metadata,match)
    paths={'candidate':checkpoint}
    if reference_checkpoint is not None:paths['reference']=reference_checkpoint
    if secondary_checkpoint is not None:paths['secondary']=secondary_checkpoint
    frozen={name:load_frozen(path) for name,path in paths.items()}
    out.mkdir(parents=True)
    protocol={'status':'all_comparators_frozen_before_test','match':identity,
              'checkpoints':{name:{'sha256':digest(path),'version':frozen[name]['version'],
                              'penalty':frozen[name]['penalty'],'min_frames':frozen[name]['min_frames'],
                              'strength':frozen[name]['strength']} for name,path in paths.items()},
              'baseline_window':50,'dropout_seed':300,'selection_on_this_match':False,
              'training_on_this_match':False,'competition_data_used':False,
              'training_match':'J03WOH','validation_match':'J03WPY',
              'previously_viewed_tests':['J03WMX','J03WN1'] if match=='J03WQQ' else ['J03WMX'],
              'note':'All listed candidates and comparators are declared before any test predictions or scoring. No parameter sweep.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    data=load_sportec(tracking,match,out.parent/'cache')
    rows=conditions([(period_arrays(data,i),data['pitch']) for i in (0,1)],300)
    results={'baseline_smooth50':evaluate(rows,[baseline_prediction(s,50,mask) for s,p,c,mask in rows])}
    diagnostics={}
    for name,state in frozen.items():
        predictions=[infer_frozen(state,s,p,mask) for s,p,c,mask in rows]
        labels=[a for a,b in predictions]
        results[name+'_decoded']=evaluate(rows,labels)
        diagnostics[name]=spell_diagnostics(rows,labels)
        if name=='candidate':results['candidate_raw']=evaluate(rows,[np.argmax(b,axis=1) for a,b in predictions])
    for result in results.values():result['mean_clean_masked_spell_f1']=result.pop('selection_spell_f1')
    report={'protocol':protocol,'tracking_sha256':str(data['source_sha256']),
            'test_results':results,'diagnostics':diagnostics,
            'limitations':['One new match; no competition score.',
                           'New match J03WQQ shares its home club with training; this is not unseen-team evaluation.' if match=='J03WQQ' else 'One additional external match.',
                           'Short-prediction counts are flicker proxies, not proof that each brief prediction is false.',
                           'This match has now been viewed; future tuning requires a different independent check.',
                           'All decoding is offline and uses future evidence.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('tracking','metadata','checkpoint','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path)
    p.add_argument('--secondary-checkpoint',type=Path)
    p.add_argument('--match',choices=list(FRESH_MATCHES),default='J03WN1')
    a=p.parse_args();r=run(a.tracking,a.metadata,a.checkpoint,a.out,a.reference_checkpoint,a.match,a.secondary_checkpoint)
    print(json.dumps(r['test_results'],indent=2))
if __name__=='__main__':main()
