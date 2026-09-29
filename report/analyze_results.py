"""Rebuild the report's evidence tables and vector plot data from saved logs.

Run from any directory with Python + numpy + matplotlib. This script does not
train models, modify the submission code, or select a checkpoint using test data.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / 'runs'
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
metrics = {p.parent.name: json.loads(p.read_text())
           for p in sorted(RUNS.glob('*/metrics.json'))}
rows = []
for name, m in metrics.items():
    vals = m.get('validations', {'final': m.get('validation', {})})
    efile = RUNS / name / 'test_cpu_fp32.json'
    test = json.loads(efile.read_text()) if efile.exists() else {}
    rows.append(dict(run=name, steps=m.get('steps', m['train_tokens']//8192),
                     targets=m['train_tokens'], parameters=m['parameters'],
                     cache=m['config'].get('cache', False), kd=m.get('kd_alpha'),
                     validation_final=vals.get('final',{}).get('bpb'),
                     validation_ema=vals.get('ema',{}).get('bpb'),
                     validation_avg=vals.get('avg',{}).get('bpb'),
                     test_bpb=test.get('bpb'), test_seconds=test.get('seconds'),
                     training_seconds=m['train_seconds'],
                     implementation_sha256=m['implementation_sha256']))
with (HERE / 'experiment_summary.csv').open('w', newline='', encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.18,'savefig.dpi':220})
colors=['#286495','#cf7539']
fig, axes=plt.subplots(1,2,figsize=(9.4,3.25),sharey=True)
for ax,names,title in zip(axes,[['s_direct','s_kd'],['s_dir48','s_both48']],
                         ['Earlier recipe (48,000 steps)','Revised recipe (72,000 steps)']):
    for name,label,color in zip(names,['Direct CE','Distillation'],colors):
        h=metrics[name]['validation_history']
        x=np.array([z['step'] for z in h])/1000;y=np.array([z['bpb'] for z in h])
        ax.plot(x,y,color=color,label=label,lw=1.8)
    ax.set(title=title,xlabel='Training updates (thousands)',ylim=(1.47,1.74))
    ax.legend(frameon=False)
axes[0].set_ylabel('Validation BPB (lower is better)')
fig.tight_layout();fig.savefig(FIG/'validation_curves.png');plt.close(fig)

names=['ab_main4800','ab_gelu4800','ab_drop4800','ab_untied4800']
labels=['Reference','GELU','No dropout','Untied']
fig,ax=plt.subplots(figsize=(5.7,2.9))
x=np.arange(len(names));width=.23
for i,(variant,color) in enumerate(zip(['final','ema','avg'],['#286495','#82a99b','#cf7539'])):
    y=[metrics[n]['validations'][variant]['bpb'] for n in names]
    ax.bar(x+(i-1)*width,y,width,color=color,label=variant.upper())
ax.set(xticks=x,xticklabels=labels,ylabel='Validation BPB',ylim=(1.54,1.735))
ax.legend(ncol=3,frameon=False,loc='upper left');fig.tight_layout()
fig.savefig(FIG/'architecture_ablation.png');plt.close(fig)

checks=[]
for a,b in [('arm_no19k','arm_cache19k'),('arm_kd19k','arm_both19k'),
            ('ab_main4800','ab_nocache4800')]:
    A,B=metrics[a],metrics[b]
    checks.append(dict(pair=[a,b],identical_training_losses=
                       [v['loss'] for v in A['history']]==[v['loss'] for v in B['history']],
                       identical_validation_bpb=
                       [v['bpb'] for v in A['validation_history']]==
                       [v['bpb'] for v in B['validation_history']]))
a=np.load(RUNS/'s_dir48/test_cpu_fp32.window-nll.npy')
b=np.load(RUNS/'s_both48/test_cpu_fp32.window-nll.npy')
audit=dict(cache_pairs=checks, recorded_train_hours=sum(m['train_seconds'] for m in metrics.values())/3600,
           recorded_process_hours=sum(m.get('process_seconds',0) for m in metrics.values())/3600,
           recorded_training_targets=sum(m['train_tokens'] for m in metrics.values()),
           paired_test_windows=len(a),windows_improved=int((b<a).sum()),
           paired_test_bpb_improvement=float((a-b).sum()/np.log(2)/1292013),
           current_source_sha256={p:hashlib.sha256((ROOT/'code'/p).read_bytes()).hexdigest()
                                  for p in ['student.py','evaluate.py']})
(HERE/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
