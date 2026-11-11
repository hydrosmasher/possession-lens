"""Frozen evaluation on J03WN1, separate from all training and tuning commands."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
from .external import FRESH_MATCHES,load_sportec,period_arrays
from .pipeline import digest
from .model import instantaneous_features,decode
from .rnn import observation_probabilities,RNN_VERSION,RNN_REFINED_VERSION,RNN_SYMMETRIC_VERSION
from .rnn_benchmark import conditions,evaluate
from .experiment import baseline_prediction

def verify_metadata(path,match='J03WN1'):
    if match not in FRESH_MATCHES:raise ValueError('Not a designated fresh external match')
    root=ET.parse(path).getroot();general=root.find('.//General')
    spec=FRESH_MATCHES[match]
    if general is None or general.get('MatchId')!='DFL-MAT-'+match:raise ValueError('Metadata match identity mismatch')
    if general.get('HomeTeamId')!=spec['home'] or general.get('GuestTeamId')!=spec['away']:raise ValueError('Metadata home/away roles mismatch')
    if general.get('CompetitionId')!='DFL-COM-'+spec['competition']:raise ValueError('Metadata competition mismatch')
    return {'match_id':match,'title':general.get('MatchTitle'),'metadata_sha256':digest(path)}

def run(tracking,metadata,checkpoint,out,reference_checkpoint=None):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output folder; a repeated viewed test is not fresh evidence')
    identity=verify_metadata(metadata)
    with np.load(checkpoint,allow_pickle=False) as d:
        if str(d['version']) not in (RNN_VERSION,RNN_REFINED_VERSION,RNN_SYMMETRIC_VERSION):raise ValueError('Incompatible checkpoint')
        model={k:d[k] for k in ('mean','scale','wx','wh','bh','wy','by')}
        team=float(d['transition_penalty']);stop=float(d['stop_penalty']) if 'stop_penalty' in d.files else team
        symmetric=str(d['version'])==RNN_SYMMETRIC_VERSION
        if symmetric and ('symmetric' not in d.files or not bool(d['symmetric'])):raise ValueError('Missing symmetry setting')
    if not np.isfinite([team,stop]).all() or min(team,stop)<0:raise ValueError('Invalid decoder settings')
    reference=None;reference_penalty=None
    if reference_checkpoint is not None:
        with np.load(reference_checkpoint,allow_pickle=False) as d:
            if str(d['version'])!=RNN_VERSION:raise ValueError('Reference must be the original tested RNN')
            reference={k:d[k] for k in ('mean','scale','wx','wh','bh','wy','by')}
            reference_penalty=float(d['transition_penalty'])
        if not np.isfinite(reference_penalty) or reference_penalty<0:raise ValueError('Invalid reference penalty')
    out.mkdir(parents=True)
    protocol={'status':'frozen_before_fresh_test','match':identity,'model_sha256':digest(checkpoint),
              'team_penalty':team,'stop_penalty':stop,'team_symmetric':symmetric,'baseline_window':50,'dropout_seed':300,
              'selection_on_this_match':False,'training_on_this_match':False,'competition_data_used':False,
              'training_match':'J03WOH','validation_match':'J03WPY','previously_viewed_test':'J03WMX',
              'reference_checkpoint_sha256':digest(reference_checkpoint) if reference is not None else None,
              'reference_penalty':reference_penalty,
              'protocol_note':'Fixed smooth50 baseline, raw/decoded candidate, and optional frozen current model. No parameter sweep or selection.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    data=load_sportec(tracking,'J03WN1',out.parent/'cache')
    parts=[(period_arrays(data,i),data['pitch']) for i in (0,1)]
    rows=conditions(parts,300)
    probs=[observation_probabilities(model,s,p,symmetric=symmetric,ball_missing=mask) for s,p,c,mask in rows]
    results={'baseline_smooth50':evaluate(rows,[baseline_prediction(s,50,mask) for s,p,c,mask in rows]),
             'rnn_raw':evaluate(rows,[np.argmax(p,axis=1) for p in probs]),
             'rnn_decoded':evaluate(rows,[decode(p,team,stop_penalty=stop) for p in probs])}
    if reference is not None:
        ref_probs=[observation_probabilities(reference,s,p,ball_missing=mask) for s,p,c,mask in rows]
        results['reference_decoded']=evaluate(rows,[decode(p,reference_penalty) for p in ref_probs])
    # evaluate() shares its implementation with development scoring; this mean is descriptive here, never a selection criterion.
    for result in results.values():result['mean_clean_masked_spell_f1']=result.pop('selection_spell_f1')
    report={'protocol':protocol,'tracking_sha256':str(data['source_sha256']),
            'test_results':results,'limitations':['One additional external match; no competition score.',
            'Synthetic ball dropout does not model all real tracking errors.',
            'This match has now been viewed; further tuning requires another independent test.',
            'Offline Viterbi uses future evidence.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('tracking','metadata','checkpoint','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path)
    a=p.parse_args();r=run(a.tracking,a.metadata,a.checkpoint,a.out,a.reference_checkpoint)
    print(json.dumps(r['test_results'],indent=2))
if __name__=='__main__':main()
