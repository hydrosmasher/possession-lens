"""Conservative anonymous identity hypotheses, calibrated on external tracks only.

External PersonId values are used solely to score simulated fragment linking.
No player names, possession labels, shirt numbers or video are inputs to linking.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
from .external import ALL_MATCHES,period_arrays
from .pipeline import digest


def external_identities(xml,cache,match):
    with np.load(cache,allow_pickle=False) as d:data={k:d[k] for k in d.files}
    if str(data['match_id'])!=match or str(data['source_sha256'])!=digest(xml):raise ValueError('External identity source/cache mismatch')
    spec=ALL_MATCHES[match];keys=[];current=None;count=0
    for event,e in ET.iterparse(xml,events=('start','end')):
        if event=='start' and e.tag=='FrameSet':current=(e.get('GameSection'),e.get('TeamId'),e.get('PersonId'));count=0
        elif event=='end' and e.tag=='Frame':count+=1;e.clear()
        elif event=='end' and e.tag=='FrameSet':
            if count and current[1] in (spec['home'],spec['away']):keys.append(current)
            current=None;e.clear()
    parts=[]
    for i,period in enumerate(('firstHalf','secondHalf')):
        s=period_arrays(data,i);ids=[(team,person) for p,team,person in keys if p==period]
        if len(ids)!=s['xy'].shape[1] or any(not person for team,person in ids):raise ValueError('External PersonId alignment failed')
        if len(set(person for team,person in ids))!=len(ids):raise ValueError('Duplicate external identity')
        if not np.array_equal(s['teams'],[0 if team==spec['home'] else 1 for team,person in ids]):raise ValueError('External identity team order differs from cache')
        parts.append((s['xy'],s['teams'],[person for team,person in ids]))
    return parts


def motion_limit(xy):
    speed=np.linalg.norm(np.diff(xy,axis=0),axis=2)*25
    valid=speed[np.isfinite(speed)]
    if not len(valid):raise ValueError('No valid external movement for calibration')
    return float(max(1.,1.5*np.quantile(valid,.999)))


def fragments(xy,teams,jump_limit,truth=None,simulate=False,seed=42):
    """Split observed spans at gaps/jumps. Simulation adds label-independent cuts."""
    rng=np.random.default_rng(seed);out=[]
    for slot,team in enumerate(teams):
        valid=np.flatnonzero(np.isfinite(xy[:,slot]).all(axis=1))
        if not len(valid):continue
        separated=(np.diff(valid)>1)|(np.linalg.norm(np.diff(xy[valid,slot],axis=0),axis=1)*25>jump_limit)
        cuts=np.r_[0,np.flatnonzero(separated)+1,len(valid)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            a=int(valid[lo]);end=int(valid[hi-1])+1
            while a<end:
                b=min(end,a+int(rng.integers(150,751))) if simulate else end
                tail=xy[max(a,b-6):b,slot]
                velocity=np.median(np.diff(tail,axis=0),axis=0)*25 if len(tail)>1 else np.zeros(2)
                if np.linalg.norm(velocity)>jump_limit:velocity=np.zeros(2)
                head=xy[a:min(b,a+6),slot]
                initial_velocity=np.median(np.diff(head,axis=0),axis=0)*25 if len(head)>1 else np.zeros(2)
                if np.linalg.norm(initial_velocity)>jump_limit:initial_velocity=np.zeros(2)
                out.append({'slot':int(slot),'team':int(team),'start':a,'end':b,'first':xy[a,slot].copy(),
                            'last':xy[b-1,slot].copy(),'velocity':velocity,'initial_velocity':initial_velocity,
                            **({'truth':truth[slot]} if truth is not None else {})})
                a=b+int(rng.integers(1,26)) if simulate else end
    return sorted(out,key=lambda f:(f['start'],f['team'],f['slot']))


def link_fragments(frags,max_gap=25,max_error=1.,ambiguity_margin=.5):
    """Greedy motion linking with team, gap, ambiguity and no-overlap constraints.

    Truth/slot identity are not used in the matching cost. New fragments can be
    left separate; the method never forces the number of groups to be 11.
    """
    if max_gap<1 or max_error<=0 or ambiguity_margin<0:raise ValueError('Invalid identity-link settings')
    tails={};assignments=[];links=[];next_id=0
    for i,f in enumerate(frags):
        candidates=[]
        for group,previous in tails.items():
            old=frags[previous];gap=f['start']-(old['end']-1)
            if old['team']!=f['team'] or old['end']>f['start'] or not 0<gap<=max_gap:continue
            expected=old['last']+old['velocity']*(gap/25)
            backward=f['first']-f.get('initial_velocity',f['velocity'])*(gap/25)
            error=float(max(np.linalg.norm(expected-f['first']),np.linalg.norm(backward-old['last'])))
            if error<=max_error:candidates.append((error,group,previous))
        candidates.sort()
        if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>=ambiguity_margin):
            error,group,previous=candidates[0];links.append((previous,i,error))
        else:group=next_id;next_id+=1
        tails[group]=i;assignments.append(group)
    return np.array(assignments,dtype=int),links


def score_links(frags,links):
    opportunities=0;last={}
    for f in frags:
        key=(f['team'],f['truth']);old=last.get(key)
        if old is not None and 0<f['start']-(old['end']-1)<=50:opportunities+=1
        last[key]=f
    correct=sum(frags[a]['truth']==frags[b]['truth'] and frags[a]['team']==frags[b]['team'] for a,b,error in links)
    return {'links':len(links),'correct_links':int(correct),'false_links':int(len(links)-correct),
            'link_precision':correct/len(links) if links else None,
            'eligible_true_links_at_50_frames':opportunities,'link_recall':correct/opportunities if opportunities else 0.}


def calibrate(train_xml,train_cache,test_xml,test_cache,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new identity calibration folder')
    # This is an identity-linking task, separate from the previously scored possession task.
    train=external_identities(train_xml,train_cache,'J03WPY')
    limit=motion_limit(train[0][0]);out.mkdir(parents=True)
    protocol={'task':'identity linking under simulated fragmentation, not player possession accuracy',
              'calibration_match':'J03WPY','calibration_period':'firstHalf','validation_period':'secondHalf',
              'held_out_identity_match':'J03WN1','simulation':'6–30 second fragments; 1–25 missing frames; no appearance/jump noise',
              'seeds':[42,43,44,45],'gap_grid':[10,25,50],'error_grid_metres':[.1,.25,.5,1.,2.],
              'ambiguity_grid_metres':[.25,.5,1.,2.],'minimum_validation_precision':.995,
              'jump_limit_mps_from_first_half':limit,'provided_match_used_for_fitting':False,
              'selection':'validation precision at least 0.995 and at least 30 links; then maximum eligible-link recall; ties fewer false links'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    train_frags=fragments(*train[0][:2],limit,truth=train[0][2],simulate=True,seed=42)
    val_frags=fragments(*train[1][:2],limit,truth=train[1][2],simulate=True,seed=43)
    trials=[]
    for gap in protocol['gap_grid']:
        for error in protocol['error_grid_metres']:
            for margin in protocol['ambiguity_grid_metres']:
                settings={'max_gap':gap,'max_error':error,'ambiguity_margin':margin}
                _,links=link_fragments(val_frags,**settings)
                trials.append({**settings,**score_links(val_frags,links)})
    eligible=[r for r in trials if r['links']>=30 and r['link_precision']>=.995]
    (out/'validation-trials.json').write_text(json.dumps(trials,indent=2)+'\n')
    if not eligible:raise ValueError('No conservative identity setting passed external validation; do not link provided tracks')
    selected=max(eligible,key=lambda r:(r['link_recall'],-r['false_links'],-r['max_error']))
    settings={k:selected[k] for k in ('max_gap','max_error','ambiguity_margin')}
    model={'version':'anonymous-motion-link-v1','settings':settings,'jump_limit_mps':limit,
           'external_only_calibration':True,'identities_verified_on_provided_match':False,
           'calibration_xml_sha256':digest(train_xml),'protocol':protocol}
    (out/'identity-model.json').write_text(json.dumps(model,indent=2)+'\n')
    # Freeze the model before loading/scoring the held-out identity match.
    test=external_identities(test_xml,test_cache,'J03WN1');test_scores=[]
    for i,(xy,teams,truth) in enumerate(test):
        f=fragments(xy,teams,limit,truth=truth,simulate=True,seed=44+i);_,links=link_fragments(f,**settings)
        test_scores.append(score_links(f,links))
    total={k:sum(s[k] for s in test_scores) for k in ('links','correct_links','false_links','eligible_true_links_at_50_frames')}
    total['link_precision']=total['correct_links']/total['links'] if total['links'] else None
    total['link_recall']=total['correct_links']/total['eligible_true_links_at_50_frames'] if total['eligible_true_links_at_50_frames'] else 0.
    report={'protocol':protocol,'validation_trials':trials,'selected_validation':selected,'held_out_identity_test':total,
            'model_sha256':digest(out/'identity-model.json'),'test_xml_sha256':digest(test_xml),
            'limitations':['Synthetic fragmentation is easier than real identity switches and wrong detections.',
                           'High link precision here does not establish provided-match identity accuracy.',
                           'No appearance or shirt-number recognition; groups are anonymous hypotheses.',
                           'External PersonIds score linking only, not ball-contact or individual possession accuracy.']}
    (out/'identity-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
