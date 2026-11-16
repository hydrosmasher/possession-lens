"""External-label counterfactual diagnostics, never deployable model outputs."""
from pathlib import Path
import json,numpy as np
from .research_snapshot import cached_case
from .control_state import evaluate_accuracy
from .tracker_robustness import CONDITIONS

def run(root):
    root=Path(root);descriptor={'source':'short-spell-053','name':'short_focus'};rows=[];preds={k:[] for k in ('frozen053','reference_dead_correction','reference_team_correction')}
    for fi in (0,1):
        for i in range(8):
            y,p,q=cached_case(descriptor,fi,i,root/'local');rows.append({'y':y,'condition':CONDITIONS[i%4]});preds['frozen053'].append(p)
            dead=p.copy();dead[y==2]=2;missing_live=(y<2)&(dead==2);dead[missing_live]=q[missing_live,:2].argmax(1);preds['reference_dead_correction'].append(dead)
            team=p.copy();known_live=(y<2)&(team<2);team[known_live]=y[known_live];preds['reference_team_correction'].append(team)
    scores={k:evaluate_accuracy(rows,p) for k,p in preds.items()};report={'purpose':'External reference-label error-budget diagnostic only; these are NOT algorithms, NOT candidate predictions, NOT achievable guarantees and NEVER eligible for deployment/selection. No challenge labels used.','reference':'frozen053','scenarios':{'reference_dead_correction':'Force correct external D/live mask; keep existing team decisions and fill removed D using existing conditional H/A argmax. Not a formal upper bound.','reference_team_correction':'Correct H/A team labels where prediction and reference are live; leave all D/live errors unchanged. Not a formal upper bound.'},'scores':scores,'deployable':False,'used_for_training':False,'independent_test':False};(root/'reports/external/diagnostic-error-budget-053.json').write_text(json.dumps(report,indent=2)+'\n')
    for k,v in scores.items():print(k,v['mean_spell_f1'],{c:round(v[c]['spell_f1_mean'],4) for c in CONDITIONS})
if __name__=='__main__':run(Path.cwd())
