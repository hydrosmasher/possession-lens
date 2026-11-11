"""Bounded validation-only test of team-permutation averaging on frozen RNN weights."""
from pathlib import Path
import argparse,json
import numpy as np
from .rnn_refine import read_cache
from .rnn_benchmark import conditions,evaluate
from .rnn import observation_probabilities,RNN_VERSION,RNN_SYMMETRIC_VERSION
from .model import decode
from .pipeline import digest

SYMMETRIC_VERSION=RNN_SYMMETRIC_VERSION

def run(training_reference,validation_cache,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new run folder')
    rows=conditions(read_cache(validation_cache,'J03WPY'),100)
    with np.load(training_reference,allow_pickle=False) as d:
        if str(d['version'])!=RNN_VERSION:raise ValueError('Expected original frozen recurrent weights')
        model={k:d[k] for k in ('mean','scale','wx','wh','bh','wy','by')}
    out.mkdir(parents=True)
    protocol={'validation_only':True,'validation_match':'J03WPY','penalties':[6,12,24,48],
              'new_training':False,'source_model_sha256':digest(training_reference),
              'operation':'average original and team-swapped RNN probabilities with output H/A realigned',
              'J03WMX_loaded':False,'competition_data_used':False,'fresh_test_required':True}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    trials=[]
    for symmetric in (False,True):
        probs=[observation_probabilities(model,s,p,symmetric,mask) for s,p,c,mask in rows]
        for penalty in protocol['penalties']:
            trials.append({'symmetric':symmetric,'penalty':penalty,**evaluate(rows,[decode(p,penalty) for p in probs])})
    selected=max(trials,key=lambda t:t['selection_spell_f1'])
    np.savez_compressed(out/'candidate-model.npz',**model,
                        version=np.array(SYMMETRIC_VERSION if selected['symmetric'] else RNN_VERSION),
                        symmetric=np.array(selected['symmetric']),decoder=np.array('viterbi'),
                        transition_penalty=np.array(selected['penalty']))
    report={'protocol':protocol,'selected_validation':selected,'validation_trials':trials,
            'candidate_sha256':digest(out/'candidate-model.npz'),'test_results':None,
            'limitations':['Validation reused after earlier experiments; fresh independent evidence required.',
                           'Team symmetry averaging doubles recurrent inference cost.','Offline decoder uses future observations.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--validation-cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=run(a.reference,a.validation_cache,a.out);print(json.dumps(r['selected_validation'],indent=2))
if __name__=='__main__':main()
