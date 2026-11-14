"""Optional figure regeneration: requires Matplotlib; no match data read."""
from pathlib import Path
import os,json
r=Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(r/'local/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
names=['019','030','031','032','037','038','040','043'];scores=[.46259620366529636,.5047756536725319,.522219097386216,.5362134387321524,.5588328790376739,.5625634880544811,.5855742496696709,.6091880683877775];report=json.loads((r/'reports/external/motion-gru-043-report.json').read_text());old=json.loads((r/'reports/external/ranking-030-report.json').read_text())['current'];new=report['selected']['scores'];conditions=['clean','masked','long_gap','tracker_noise'];labels=['Clean','Short gaps','Long gaps','Tracking noise']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');ax=axes[0];ax.plot(names,scores,color='#165ac7',marker='o',lw=2)
for i,v in enumerate(scores):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
ax.axhline(.7,color='#a71948',linestyle='--',lw=1.2,label='0.7 goal');ax.set_ylim(.4,.73);ax.set_ylabel('Official possession-spell F1');ax.set_xlabel('Frozen candidate');ax.set_title('Comparable development mean');ax.grid(axis='y',alpha=.2);ax.legend(loc='lower right',frameon=False)
ax=axes[1];x=list(range(4));ax.bar([i-.18 for i in x],[old[c]['spell_f1_mean'] for c in conditions],width=.36,label='019',color='#b7bec9');ax.bar([i+.18 for i in x],[new[c]['spell_f1_mean'] for c in conditions],width=.36,label='043',color='#a71948');ax.set_xticks(x,labels,rotation=15);ax.set_ylim(0,.8);ax.set_title('Conditions behind the mean');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
fig.suptitle('Possession Lens · external development progress',fontsize=15,fontweight='bold');fig.supxlabel('Two heavily reused matches; simulated gaps/noise. Not independent test or competition results.',fontsize=8);p=r/'reports/external/progress-043.png';fig.savefig(p,dpi=180);plt.close(fig)
(r/'reports/external/progress-043.json').write_text(json.dumps({'candidates':names,'mean_spell_f1':scores,'same_original_external_condition_draws':True,'independent_test':False,'competition_percentile_known':False},indent=2)+'\n');print(p)
