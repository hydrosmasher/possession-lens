"""026: honest low-confidence abstention under missing ball observations."""
from pathlib import Path
import argparse,json
import numpy as np
from .tracker_robustness import CONDITIONS,interval_mask
from .experiment import missing_mask,aggregate
from .gap_possession import long_mask,diagnostics
from .pipeline import pm,digest

VERSION='uncertainty-had-026'

def abstain(labels,p,observed,threshold):
    labels=np.asarray(labels);p=np.asarray(p);observed=np.asarray(observed)
    if labels.ndim!=1 or p.shape!=(len(labels),3) or observed.shape!=labels.shape or not np.isin(labels,[0,1,2]).all() or not np.isfinite(p).all() or (p<0).any() or not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid abstention inputs')
    margin=np.abs(p[:,0]-p[:,1])/np.maximum(p[:,:2].sum(1),1e-12)
    out=labels.copy();out[(labels<2)&~observed.astype(bool)&(margin<threshold)]=3
    return out

def evaluate(rows,preds,current=None):
    result={}
    for c in CONDITIONS:
        ix=[i for i,r in enumerate(rows) if r['condition']==c];score=aggregate([(rows[i]['y'],preds[i]) for i in ix]);details=[]
        for i in ix:
            y=rows[i]['y'];effective=pm.fill_abstentions(preds[i],pm.valid_runs(y));details.append(diagnostics(y,effective,current[i] if current is not None else None))
        score['diagnostics']={k:sum(d[k] for d in details) for k in details[0]};y=np.concatenate([rows[i]['y'] for i in ix]);p=np.concatenate([preds[i] for i in ix]);tp=int(((y==2)&(p==2)).sum());score['dead_f1']=2*tp/max(int((y==2).sum()+(p==2).sum()),1);score['abstained_fraction']=float((p==3).mean());result[c]=score
    result['mean_spell_f1']=float(np.mean([result[c]['spell_f1_mean'] for c in CONDITIONS]));result['totals']={k:sum(result[c]['diagnostics'][k] for c in CONDITIONS) for k in result['clean']['diagnostics']}
    return result

def run(previous,out):
    previous=Path(previous);out=Path(out)
    if out.exists():raise ValueError('Choose a new folder')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'history':'Follow-up after020–024 on same two extensively reused matches; no independent test','competition_data_used':False,'input_protocol_sha256':digest(previous/'protocol.json'),'frozen':'020 balanced_half decoder on unchanged019 emissions; head and weights fixed','thresholds':[.1,.2],'abstention_rule':'X only for missing ball, predicted H/A and conditional H/A probability margin below threshold. D decisions retained. No abstention from real ball observations.','missingness':'Exact original short/long mask seeds214/414 and tracker-noise first RNG mask818+i; source Sportec ball fully observed before simulation','metric_policy':'Official spellF1 fills X within each reference run; diagnostics use exactly that fill; frame macroF1 uses raw X and charges misses. No interpolation of reference labels.','selection':'Higher four-condition meanF1; no total misses/merges/FP increase; prior condition fold loss<=.01; noise fold no loss; deadF1 decline<=.005; macroF1 loss<=.02 and abstention fraction<=.05 for every fold/condition. No post-result threshold sweep.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[];current=[];preds={v:[] for v in (.1,.2)};folds=[]
    for fold in (0,1):
        fr=[];fc=[];fp={v:[] for v in preds};fd=out/f'fold-{fold}';fd.mkdir()
        for half in (0,1):
            for ci,c in enumerate(CONDITIONS):
                i=half*4+ci
                with np.load(previous/f'fold-{fold}/validation-{i}.npz',allow_pickle=False) as a:y=a['reference'];p=a['emissions'];base=a['current'];candidate=a['balanced_half']
                n=len(y);mask=np.zeros(n,bool)
                if c=='masked':mask=missing_mask(n,214+half)
                elif c=='long_gap':mask=long_mask(n,414+half)
                elif c=='tracker_noise':mask=interval_mask(n,np.random.default_rng(818+half),.4,50,300)
                fr.append({'y':y,'condition':c});fc.append(base)
                for t in preds:fp[t].append(abstain(candidate,p,~mask,t))
                np.savez_compressed(fd/f'validation-{i}.npz',reference=y,current=base,**{'threshold'+str(t):a[-1] for t,a in fp.items()})
        r={'validation':'J03WN1' if fold==0 else 'J03WMX','current':evaluate(fr,fc),'variants':{str(t):evaluate(fr,p,fc) for t,p in fp.items()}};folds.append(r);rows+=fr;current+=fc
        for t in preds:preds[t]+=fp[t]
        (fd/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    base=evaluate(rows,current);trials=[]
    for t in preds:
        score=evaluate(rows,preds[t],current);a=score['totals'];b=base['totals'];guards={'mean_f1':score['mean_spell_f1']>base['mean_spell_f1'],'misses':a['missed']<=b['missed'],'merges':a['merges']<=b['merges'],'false_positives':a['false_positive']<=b['false_positive'],'prior_conditions':all(f['variants'][str(t)][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'tracker_noise':all(f['variants'][str(t)]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),'dead_f1':all(f['variants'][str(t)][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS),'macro_f1':all(f['variants'][str(t)][c]['macro_f1']>=f['current'][c]['macro_f1']-.02 for f in folds for c in CONDITIONS),'abstention_cap':all(f['variants'][str(t)][c]['abstained_fraction']<=.05 for f in folds for c in CONDITIONS)}
        trials.append({'threshold':t,'scores':score,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];report={'protocol':protocol,'folds':folds,'current':base,'trials':trials,'selected':max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None,'fresh_test_results':None,'promoted_to_default':False,'limitations':['Same heavily reused match labels; no fresh test.','Abstentions repeat previous states for spell scoring and may delay real turnovers.','Raw frame macroF1 is charged for X.','No supplied match labels/video/tracking or public score used.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('CURRENT',base['mean_spell_f1'],base['totals'])
    for t in trials:print(t['threshold'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'])
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.previous,a.out)
if __name__=='__main__':main()
