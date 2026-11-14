"""Follow-up019: combine frozen dead-state experts, preserving017 team control."""
from pathlib import Path
import argparse,json
import numpy as np
from .tracker_robustness import CONDITIONS,condition,evaluate
from .external import period_arrays
from .rnn_runtime import load_frozen,infer_frozen
from .control_correction import corrected_probabilities,blend_control
from .deadball import observation_features,fuse
from .model import decode
from .pipeline import digest
from . import tree_model

VERSION='dead-expert-blend-019'

def combine(left,right,mix):
    a=left['meta'];b=right['meta']
    if a['kind']!='hist_boost' or b['kind']!='hist_boost' or a['features']!=b['features'] or a['classes']!=b['classes'] or not np.isfinite(mix) or not 0<=mix<=1 or a['count']+b['count']>512:raise ValueError('Incompatible frozen experts or blend')
    meta={**a,'count':a['count']+b['count'],'baseline':(1-mix)*a['baseline']+mix*b['baseline']}
    trees=[]
    for model,weight in [(left,1-mix),(right,mix)]:
        for t in model['trees']:
            copy={k:v.copy() for k,v in t.items()};copy['value']*=weight;trees.append(copy)
    return {'meta':meta,'trees':trees}

def save_combined(path,model):
    arrays={'meta':np.array(json.dumps(model['meta']))}
    for i,t in enumerate(model['trees']):arrays.update({str(i)+'_'+k:v for k,v in t.items()})
    np.savez_compressed(path,**arrays)

def run(cache_root,reference_root,forward_dead,reverse_dead,noise_root,team_model,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new experiment directory')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'history':'Follow-up after viewing018. Same two extensively reused Bundesliga matches, no independent testing.','competition_data_used':False,'sources':'Existing first-division Bundesliga IDSSE/Sportec CC BY4.0; frozen017 H/A correction,014/015 old dead experts and018 noise-trained dead experts','conditions':list(CONDITIONS),'noise':'Exactly018 generator, validation seed818+i; previous short/long seeds214/414+i','mixes':[.25,.5],'fusion':'Convex average of frozen old/new dead-expert log odds; represented exactly as one200-tree numeric NPZ. Team control and decoder unchanged.','selection':'Highest four-condition mean spellF1 with no total missed/merged/FP increase; each fold original3 conditions no F1 decline>.01; tracker_noise each fold no F1 decline; every deadF1 decline<=.005. No post-result sweep.','frozen_decoder':{'mix':.75,'penalty':12,'stop_penalty':3,'dead_weight':.75},'reference_hashes':{'team':digest(team_model),'old_dead':[digest(forward_dead),digest(reverse_dead)],'noise_dead':[digest(Path(noise_root)/f'fold-{i}/dead-estimator.npz') for i in (0,1)]}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');backbone=load_frozen(team_model);all_rows=[];all_current=[];all_preds={v:[] for v in protocol['mixes']};folds=[]
    for fold,(val,old_path) in enumerate([('J03WN1',forward_dead),('J03WMX',reverse_dead)]):
        fd=out/f'fold-{fold}';fd.mkdir();paths=list(Path(cache_root).glob(val+'-*.npz'))
        if len(paths)!=1:raise ValueError('Ambiguous external cache')
        with np.load(paths[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
        if str(record['match_id'])!=val:raise ValueError('Match identity mismatch')
        old=tree_model.load(old_path);new=tree_model.load(Path(noise_root)/f'fold-{fold}/dead-estimator.npz');control=tree_model.load(Path(reference_root)/f'fold-{fold}/control-estimator.npz')
        combined={mix:combine(old,new,mix) for mix in protocol['mixes']}
        for mix,model in combined.items():
            p=fd/f'dead-blend-{mix}.npz';save_combined(p,model);combined[mix]=tree_model.load(p)
        rows=[];current=[];preds={mix:[] for mix in protocol['mixes']}
        for half in (0,1):
            original=period_arrays(record,half);pitch=record['pitch']
            for c in CONDITIONS:
                print('Validation',fold,half,c,flush=True);s,mask=condition(original,c,half,False);rows.append({'y':original['y'],'condition':c})
                _,p=infer_frozen(backbone,s,pitch,mask);h=blend_control(p,corrected_probabilities(control,s,pitch,mask),.75);x=observation_features(s,pitch,mask)
                old_d=tree_model.predict(old,x)[:,1];current.append(decode(fuse(h,old_d,.75),12,stop_penalty=3))
                for mix,model in combined.items():preds[mix].append(decode(fuse(h,tree_model.predict(model,x)[:,1],.75),12,stop_penalty=3))
        result={'validation':val,'current':evaluate(rows,current),'variants':{str(mix):evaluate(rows,p,current) for mix,p in preds.items()}}
        folds.append(result);(fd/'report.json').write_text(json.dumps(result,indent=2)+'\n');all_rows+=rows;all_current+=current
        for mix in all_preds:all_preds[mix]+=preds[mix]
        for i,r in enumerate(rows):np.savez_compressed(fd/f'validation-{i}.npz',reference=r['y'],current=current[i],**{'mix'+str(mix):p[i] for mix,p in preds.items()})
    current=evaluate(all_rows,all_current);trials=[]
    for mix in protocol['mixes']:
        score=evaluate(all_rows,all_preds[mix],all_current);a=score['totals'];b=current['totals'];guards={'mean_f1':score['mean_spell_f1']>current['mean_spell_f1'],'misses':a['missed']<=b['missed'],'merges':a['merges']<=b['merges'],'false_positives':a['false_positive']<=b['false_positive'],'prior_conditions':all(f['variants'][str(mix)][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'tracker_noise':all(f['variants'][str(mix)]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),'dead_f1':all(f['variants'][str(mix)][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS)}
        trials.append({'dead_expert_mix':mix,'scores':score,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];selected=max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'best_unrestricted':max(trials,key=lambda t:t['scores']['mean_spell_f1']),'fresh_test_results':None,'promoted_to_default':False,'limitations':['Selection follows018 on the same repeatedly used provider-labelled matches; no fresh generalisation claim.','Synthetic errors cannot represent all challenge tracker failures.','Diagnostics sum repeated physical possessions across stress conditions.','No provided game data, labels, video or public score used.','Offline context uses future tracking.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print('MIX',t['dead_expert_mix'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','reference-root','forward-dead','reverse-dead','noise-root','team-model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,a.reference_root,a.forward_dead,a.reverse_dead,a.noise_root,a.team_model,a.out)
if __name__=='__main__':main()
