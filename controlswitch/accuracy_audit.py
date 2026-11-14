"""027 frame-accuracy/error audit of frozen019 external predictions, no fitting."""
from pathlib import Path
import argparse,json
import numpy as np
from .pipeline import pm,digest
from .experiment import aggregate
from .tracker_robustness import CONDITIONS

def frame_report(y,p):
    y=np.asarray(y);p=np.asarray(p);known=y<3
    if y.shape!=p.shape or y.ndim!=1 or not np.isin(y,[0,1,2,3]).all() or not np.isin(p,[0,1,2,3]).all():raise ValueError('Invalid labels')
    if not known.any():raise ValueError('Accuracy requires labelled external frames')
    cm=np.zeros((3,4),int);np.add.at(cm,(y[known],p[known]),1)
    idx=np.arange(len(y));boundaries=np.flatnonzero(y[1:]!=y[:-1])+1
    if len(boundaries):
        prev=np.searchsorted(boundaries,idx,side='right')-1;nxt=np.minimum(prev+1,len(boundaries)-1);distance=np.minimum(np.where(prev>=0,np.abs(idx-boundaries[np.maximum(prev,0)]),len(y)),np.abs(idx-boundaries[nxt]))
    else:distance=np.full(len(y),len(y))
    wrong=(y!=p)&known;near=known&(distance<=25);interior=known&~near
    return {'n_scored':int(known.sum()),'correct':int((known&~wrong).sum()),'accuracy':float(np.mean(y[known]==p[known])),'confusion_rows_HAD_cols_HADX':cm.tolist(),'per_class_f1':{name:float(2*cm[k,k]/max(cm[k,:].sum()+cm[:,k].sum(),1)) for k,name in enumerate('HAD')},'errors':{'wrong_live_team':int(((y<2)&(p<2)&wrong).sum()),'missed_dead_ball':int(((y==2)&(p<2)).sum()),'false_dead_ball':int(((y<2)&(p==2)).sum()),'abstention':int((known&(p==3)).sum())},'timing':{'near_transition_frames':int(near.sum()),'near_transition_errors':int((wrong&near).sum()),'interior_frames':int(interior.sum()),'interior_errors':int((wrong&interior).sum())}}

def run(prediction_root,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new audit folder')
    out.mkdir(parents=True);rows=[];pairs={c:[] for c in CONDITIONS};inputs=[]
    for fold in (0,1):
        for half in (0,1):
            for ci,c in enumerate(CONDITIONS):
                f=Path(prediction_root)/f'fold-{fold}/validation-{half*4+ci}.npz'
                with np.load(f,allow_pickle=False) as a:y=a['reference'];p=a['mix0.5']
                rows.append({'validation_match':'J03WN1' if fold==0 else 'J03WMX','half':half+1,'condition':c,**frame_report(y,p)});pairs[c].append((y,p));inputs.append({'file':str(f.relative_to(prediction_root)),'sha256':digest(f)})
    pooled={}
    for c,values in pairs.items():
        y=np.concatenate([y for y,p in values]);p=np.concatenate([p for y,p in values]);r=frame_report(y,p)
        # Timing diagnostics must reset at halves, not infer transitions across concatenated periods.
        rr=[x for x in rows if x['condition']==c];r['timing']={k:sum(a['timing'][k] for a in rr) for k in rr[0]['timing']};r['spell_metrics']=aggregate(values);r['frames_to_reach_95_percent']=max(0,int(np.ceil(.95*r['n_scored']))-r['correct']);pooled[c]=r
    report={'experiment':'accuracy-audit-027','candidate':'019','target_interpretation':'95% correctly labelled frames is measurable;95th percentile among entrants requires leaderboard data and cannot be inferred here','conditions':pooled,'per_half':rows,'inputs':inputs,'training_performed':False,'competition_data_used':False,'limitations':['Two repeatedly used provider-labelled matches, not independent generalisation or official challenge accuracy.','Stress conditions repeat physical frames and are not additional matches.','Plain accuracy is prevalence-weighted; report macro/per-state F1 and ranking spellF1 alongside it.','Targets are aspirations, not guaranteed outcomes.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({c:{'accuracy':r['accuracy'],'frames_to_reach_95_percent':r['frames_to_reach_95_percent'],'errors':r['errors']} for c,r in pooled.items()},indent=2));return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prediction-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();run(a.prediction_root,a.out)
if __name__=='__main__':main()
