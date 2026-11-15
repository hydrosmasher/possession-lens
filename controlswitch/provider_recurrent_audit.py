"""056: frozen052/053 transfer diagnostic on reused partial Ligue1 labels."""
from pathlib import Path
import argparse,json
import numpy as np,torch
from .skillcorner_europe import load_match
from .recurrent_pipeline import load_pipeline,checked_path
from .gated_rnn import load_checkpoint
from .focal_refinement import infer_models
from .tracker_robustness import CONDITIONS,condition
from .control_state import evaluate_accuracy
from .pipeline import digest

def run(raw,out,root=Path.cwd()):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Choose a new diagnostic directory')
    out.mkdir(parents=True);configs={k:root/'reports/external'/n for k,n in [('052','transfer-052-config.json'),('053','short-spell-053-config.json')]}
    protocol={'version':'partial-provider-recurrent-056','match':'PSG–Marseille2269,Ligue1,existing historical MIT SkillCorner files','history':'Previously reused validation match; NOT fresh or independent. No new download.','models':{k:digest(p) for k,p in configs.items()},'purpose':'Frozen different-provider compatibility only; NO fitting,tuning,re-selection or promotion from this report.','supervision':'Partial team-possession groups; unknown remainsX. No verified live/dead reference. Censored valid runs change spell objects. Partial spellF1 and team-label agreement cannot be equated to full Sportec H/A/D benchmark or challenge accuracy. D predictions on known-team frames are counted without calling them verified false-dead errors.','conditions':list(CONDITIONS),'competition_data_used':False};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);parts,metadata=load_match(raw,'2269');b=load_pipeline(configs['052'],root);c=json.loads(configs['053'].read_text());assert c['base_config']==b['config']['base_config'] and c['gru_mix']==.5;grus={'052':b['gru'],'053':load_checkpoint(checked_path(root,c['gru']))};rows=[];preds={k:[] for k in grus}
    for half,(seq,pitch) in enumerate(parts):
        if not np.isin(seq['y'],[0,1,3]).all():raise ValueError('Expected only partial H/A/X supervision')
        for ci,c in enumerate(CONDITIONS):
            print('Frozen provider diagnostic',half,c,flush=True);s,mask=condition(seq,c,half,False);p=infer_models(b['base'],grus,{k:v for k,v in s.items() if k!='y'},pitch,mask);rows.append({'y':seq['y'],'condition':c})
            for k in grus:preds[k].append(p[k])
            np.savez_compressed(out/f'half-{half}-{c}.npz',reference=seq['y'],**p)
    scores={}
    for k,pp in preds.items():
        aggregate=evaluate_accuracy(rows,pp);score={'mean_partial_spell_f1':aggregate['mean_spell_f1'],'conditions':{}}
        for c in CONDITIONS:
            ix=[i for i,x in enumerate(rows) if x['condition']==c];known=sum(int((rows[i]['y']<2).sum()) for i in ix);live=sum(int(((rows[i]['y']<2)&(pp[i]<2)).sum()) for i in ix);correct=sum(int(((rows[i]['y']<2)&(pp[i]==rows[i]['y'])).sum()) for i in ix);dead=sum(int(((rows[i]['y']<2)&(pp[i]==2)).sum()) for i in ix)
            score['conditions'][c]={'partial_spell_f1':aggregate[c]['spell_f1_mean'],'known_team_label_frames':known,'known_team_label_agreement':correct/max(known,1),'selective_team_label_agreement':correct/max(live,1),'live_decision_coverage':live/max(known,1),'dead_predictions_on_known_team_frames':dead,'verified_dead_f1':None}
        scores[k]=score;print('PARTIAL PROVIDER',k,score['mean_partial_spell_f1'],flush=True)
    report={'protocol':protocol,'scores':scores,'match_metadata':metadata,'independent_test':False,'deployment_decision_from_this_report':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n');return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.raw,a.out)
