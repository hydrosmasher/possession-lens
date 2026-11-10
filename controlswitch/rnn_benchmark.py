"""External-only RNN decoder selection and optional fresh-match evaluation.

Run with python -m controlswitch.rnn_benchmark --help. No competition input is read.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from .external import MATCHES,load_sportec,period_arrays
from .experiment import crop_sequence,missing_mask,aggregate,baseline_prediction
from .model import instantaneous_features,decode
from .rnn import train,predict,RNN_VERSION
from .pipeline import digest

PENALTIES=(0,3,6,12,24,48)
WINDOWS=(0,25,50,125)

def load_records(roots,cache_dir):
    records={};sources={}
    for mid in MATCHES:
        xml=set();cached=set()
        for root in roots:
            xml.update(Path(root).rglob(f'*positions_raw_observed*{mid}.xml'))
            cached.update(Path(root).rglob(f'{mid}-*.npz'))
        if len(xml)>1:raise ValueError(f'Duplicate XML for {mid}')
        if xml:
            data=load_sportec(next(iter(xml)),mid,cache_dir)
        elif cached:
            if len(cached)>1:raise ValueError(f'Ambiguous caches for {mid}')
            path=next(iter(cached))
            with np.load(path,allow_pickle=False) as d:data={k:d[k] for k in d.files}
            if str(data['match_id'])!=mid:raise ValueError('Cache identity mismatch')
        else:continue
        records[mid]=data;sources[mid]=str(data['source_sha256'])
    return records,sources

def partitions(records,pilot):
    if all(m in records for m in MATCHES):
        return [ [(period_arrays(records[m],i),records[m]['pitch']) for i in (0,1)]
                 for m in ('J03WOH','J03WPY','J03WMX') ],'three_match'
    if not pilot or set(records)!={'J03WOH'}:
        raise ValueError('Need J03WOH training, J03WPY validation and J03WMX test; explicit --single-match-pilot permits validation-only development on J03WOH.')
    d=records['J03WOH'];s=period_arrays(d,0);cut=int(len(s['y'])*.7)
    return [[(crop_sequence(s,0,cut-100),d['pitch'])],[(crop_sequence(s,cut+100,len(s['y'])),d['pitch'])],[]],'single_match_validation_only'

def augmented_sequences(training):
    sequences=[]
    for i,(s,pitch) in enumerate(training):
        for swap in (False,True):
            y=s['y'].copy()
            if swap:
                y=np.where(y==0,1,np.where(y==1,0,y))
            for masked in (False,True):
                mask=missing_mask(len(y),42+i) if masked else None
                sequences.append((instantaneous_features(s,pitch,swap=swap,ball_missing=mask),y.copy()))
    return sequences

def conditions(parts,seed):
    return [(s,pitch,name,missing_mask(len(s['y']),seed+i) if name=='masked' else None)
            for i,(s,pitch) in enumerate(parts) for name in ('clean','masked')]

def probabilities(model,rows):
    return [predict(model,instantaneous_features(s,pitch,ball_missing=mask)) for s,pitch,name,mask in rows]

def evaluate(rows,preds):
    result={}
    for condition in ('clean','masked'):
        pairs=[(s['y'],p) for (s,_,c,_),p in zip(rows,preds) if c==condition]
        result[condition]=aggregate(pairs)
    result['selection_spell_f1']=float(np.mean([result[c]['spell_f1_mean'] for c in ('clean','masked')]))
    return result

def run(roots,out,epochs=5,single_match_pilot=False,evaluate_test=False):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output folder')
    records,sources=load_records(roots,out.parent/'cache')
    (training,validation,test),mode=partitions(records,single_match_pilot)
    if evaluate_test and mode!='three_match':raise ValueError('A fresh test requires all three external matches; previously viewed J03WOH second half is never reused.')
    out.mkdir(parents=True)
    protocol={'seed':42,'epochs':epochs,'hidden_units':24,'penalties':PENALTIES,'baseline_windows':WINDOWS,
              'selection':'mean clean/masked validation spellF1; ties choose first, lower penalty',
              'mode':mode,'sources_sha256':sources,'training_match':'J03WOH',
              'validation_match':'J03WPY' if test else 'J03WOH first half, last 30%, 200-frame purge',
              'test_match':'J03WMX' if test else None,'test_requested':evaluate_test,
              'masking':'40% ball dropout, contiguous 20-100 frame runs, validation seed 100, test seed 200',
              'decoder':'offline Viterbi, uses future evidence; not a live forecast',
              'competition_data_used':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Compare unaugmented and team-swap/dropout augmented training under the same split.
    candidates={}
    for name,sequences in [('plain',[(instantaneous_features(s,p),s['y']) for s,p in training]),
                           ('augmented',augmented_sequences(training))]:
        print(f'Training {name} RNN',flush=True)
        model,history=train(sequences,epochs=epochs)
        candidates[name]=model
        np.savez_compressed(out/f'{name}.npz',**model,version=np.array(RNN_VERSION))
        (out/f'{name}-loss.json').write_text(json.dumps(history)+'\n')
    rows=conditions(validation,100);trials=[]
    for window in WINDOWS:
        pred=[baseline_prediction(s,window,mask) for s,p,c,mask in rows]
        trials.append({'model':'baseline','window':window,**evaluate(rows,pred)})
    for name,model in candidates.items():
        probs=probabilities(model,rows)
        for penalty in PENALTIES:
            pred=[decode(p,penalty) for p in probs]
            trials.append({'model':name,'penalty':penalty,**evaluate(rows,pred)})
    chosen=max((r for r in trials if r['model']!='baseline'),key=lambda r:r['selection_spell_f1'])
    base=max((r for r in trials if r['model']=='baseline'),key=lambda r:r['selection_spell_f1'])
    model=candidates[chosen['model']]
    np.savez_compressed(out/'selected-model.npz',**model,version=np.array(RNN_VERSION),
                        decoder=np.array('viterbi'),transition_penalty=np.array(chosen['penalty']))
    report={'protocol':protocol,'validation_trials':trials,'selected_rnn':chosen,'selected_baseline':base,
            'model_sha256':digest(out/'selected-model.npz'),'test_results':None,
            'limitations':['Validation is model-selection evidence, not a fresh test.',
                           'RNN emissions use trailing observations; Viterbi decoding is offline.',
                           'External tracking quality differs from competition tracking.']}
    # Settings and checkpoint are frozen before any test probabilities or metrics are computed.
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if evaluate_test:
        rows=conditions(test,200);probs=probabilities(model,rows)
        report['test_results']={
            'rnn_raw':evaluate(rows,[np.argmax(p,axis=1) for p in probs]),
            'rnn_selected':evaluate(rows,[decode(p,chosen['penalty']) for p in probs]),
            'baseline_selected':evaluate(rows,[baseline_prediction(s,base['window'],mask) for s,_,_,mask in rows]),
            'baseline_smooth50':evaluate(rows,[baseline_prediction(s,50,mask) for s,_,_,mask in rows])}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-root',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--single-match-pilot',action='store_true')
    parser.add_argument('--evaluate-test',action='store_true')
    args=parser.parse_args()
    if args.epochs<1:parser.error('epochs must be positive')
    report=run(args.external_root,args.out,args.epochs,args.single_match_pilot,args.evaluate_test)
    for key in ('selected_rnn','selected_baseline','test_results'):print(key,json.dumps(report[key]))
if __name__=='__main__':main()
