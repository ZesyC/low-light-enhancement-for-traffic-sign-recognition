"""Extract ResNet-18 notebook evidence; never read legacy experiment outputs."""
import ast
import base64
import csv
import hashlib
import json
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/latex'
FIG = OUT / 'images'
TABLE = OUT / 'tables'
DATA = ROOT / 'report/evidence'
for folder in (FIG, TABLE, DATA):
    folder.mkdir(parents=True, exist_ok=True)
notebook_path = ROOT / 'notebooks/GTSRB_LLIE_Complete_Pipeline_Executed.ipynb'
notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
cells = notebook['cells']

def data(cell, output, kind='text/plain'):
    value = cells[cell]['outputs'][output]['data'][kind]
    return ''.join(value) if isinstance(value, list) else value

images = {'degradation': (7, 1), 'classical_default': (8, 0),
          'parameters': (9, 0), 'enhancers_selected': (14, 0),
          'subset_f1': (17, 2), 'subset_scatter': (17, 3),
          'subset_failures': (17, 4), 'training': (17, 5), 'ablation_original': (20, 2)}
for name, (cell, output) in images.items():
    (FIG / f'{name}.png').write_bytes(base64.b64decode(data(cell, output, 'image/png')))

# literal_eval rejects executable code; only the notebook's numeric infinity is replaced.
source = re.sub(r'\binf\b', '1e999', data(16, 1))
runs = ast.literal_eval(source)
assert len(runs) == 219
assert {r['n_images'] for r in runs} == {430}
assert {r['train_seed'] for r in runs} == {11, 22, 33}
assert 'ResNet' in ''.join(cells[10]['source'])
keys = ['accuracy', 'macro_f1', 'weighted_f1', 'psnr_mean', 'ssim_mean', 'latency_ms']
grouped = defaultdict(list)
for row in runs:
    grouped[row['condition'], row['method'], row['train_seed']].append(row)
per_seed = []
for (condition, method, seed), rows in grouped.items():
    assert len(rows) == (1 if condition == 'clean' else 3)
    per_seed.append(dict(condition=condition, method=method, seed=seed,
                         **{k:stats.mean(r[k] for r in rows) for k in keys}))
aggregate = []
for condition, method in sorted({(r['condition'],r['method']) for r in runs}):
    rows = [r for r in per_seed if r['condition']==condition and r['method']==method]
    result = dict(condition=condition,method=method)
    for key in keys:
        values = [r[key] for r in rows]
        result[key+'_mean'] = stats.mean(values)
        result[key+'_std'] = 0.0 if all(np.isinf(v) for v in values) else stats.stdev(values)
    aggregate.append(result)

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

write_csv(DATA/'resnet18_notebook_runs.csv', runs)
write_csv(DATA/'resnet18_notebook_aggregate.csv', aggregate)
points = {(r['condition'],r['method']):r for r in aggregate}
display_pattern = (r'^\s*\d+\s+(clean|mild|medium|severe)\s+(\w+)\s+'
                   r'([\d.]+) ± ([\d.]+)\s+([\d.]+) ± ([\d.]+)\s+'
                   r'(inf|[\d.]+)\s+([\d.]+)\s*$')
display_rows = list(re.finditer(display_pattern, data(17, 1), re.M))
assert len(display_rows) == 25
for match in display_rows:
    condition, method, acc, acc_sd, f1, f1_sd, psnr, ssim = match.groups()
    row = points[condition, method]
    checks = [('accuracy_mean',acc,100,2),('accuracy_std',acc_sd,100,2),
              ('macro_f1_mean',f1,100,2),('macro_f1_std',f1_sd,100,2),
              ('psnr_mean_mean',psnr,1,2),('ssim_mean_mean',ssim,1,4)]
    for key, value, scale, digits in checks:
        assert round(row[key]*scale, digits) == float(value), (condition,method,key)
print('Verified: all 25 aggregated rows match the notebook displayed table.')
methods = ['identity','gamma','he','clahe','retinex','zero_dce','retinexformer','diffusion']
names = dict(zip(methods,['Identity','Gamma','HE','CLAHE','MSR','Zero-DCE','Retinexformer','Diffusion']))
levels = ['mild','medium','severe']

def tex_rows(filename, rows):
    (TABLE/filename).write_text('\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n', encoding='utf-8')

tex_rows('quality.tex', [[names[m]]+[f"{points[c,m]['psnr_mean_mean']:.2f} / {points[c,m]['ssim_mean_mean']:.4f}" for c in levels] for m in methods])
tex_rows('subset_f1.tex', [[names[m]]+[f"{100*points[c,m]['macro_f1_mean']:.2f} $\\pm$ {100*points[c,m]['macro_f1_std']:.2f}" for c in levels] for m in methods])
tex_rows('latency.tex', [[names[m],f"{points['medium',m]['latency_ms_mean']:.3f}"] for m in methods])
tex_rows('subset_acc.tex', [[names[m]]+[f"{100*points[c,m]['accuracy_mean']:.2f} $\\pm$ {100*points[c,m]['accuracy_std']:.2f}" for c in levels] for m in methods])

def ranks(values):
    return np.array([1+sum(w<v for w in values)+(sum(w==v for w in values)-1)/2 for v in values])

correlations=[]
for level in levels:
    for quality in ['psnr_mean','ssim_mean']:
        for metric in ['accuracy','macro_f1']:
            x=np.array([points[level,m][quality+'_mean'] for m in methods])
            y=np.array([points[level,m][metric+'_mean'] for m in methods])
            correlations.append(dict(condition=level,quality=quality,metric=metric,n=8,
                                     pearson=float(np.corrcoef(x,y)[0,1]),
                                     spearman=float(np.corrcoef(ranks(x),ranks(y))[0,1])))
write_csv(DATA/'resnet18_correlations.csv',correlations)
level_names={'mild':'Nhẹ','medium':'Vừa','severe':'Nặng'}
tex_rows('correlations.tex',[[level_names[r['condition']], 'PSNR' if r['quality']=='psnr_mean' else 'SSIM',
                            'Accuracy' if r['metric']=='accuracy' else 'Macro-F1',
                            f"{r['pearson']:.3f}",f"{r['spearman']:.3f}"] for r in correlations])

ablation = []
for line in data(20,1).splitlines():
    match = re.match(r'\s*\d+\s+(gamma_\w+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s*$',line)
    if match:
        variant, method, mean, sd = match.groups()
        ablation.append(dict(variant=variant,method=method,macro_f1_mean=float(mean),macro_f1_std=float(sd)))
assert len(ablation)==32
write_csv(DATA/'resnet18_ablation_displayed.csv',ablation)
ap = {(r['variant'],r['method']):r for r in ablation}
variants = ['gamma_only','gamma_gaussian','gamma_poisson','gamma_both']
tex_rows('ablation.tex', [[names[m]]+[f"{100*ap[v,m]['macro_f1_mean']:.2f}" for v in variants] for m in methods])

# Full-test values are transcribed from the current README, not inferred from the subset.
readme = (ROOT/'README.md').read_text(encoding='utf-8')
full = []
for line in readme.splitlines():
    cells_ = [s.strip() for s in line.strip('|').split('|')]
    if len(cells_)==4 and cells_[0] in methods and all('±' in s for s in cells_[1:]):
        for level, text in zip(levels,cells_[1:]):
            mean,sd = map(float,text.split('±'))
            full.append(dict(method=cells_[0],condition=level,macro_f1_percent=mean,sd_percent=sd))
assert len(full)==24
write_csv(DATA/'resnet18_fulltest_readme.csv',full)
fp = {(r['condition'],r['method']):r for r in full}
tex_rows('fulltest.tex',[[names[m]]+[f"{fp[c,m]['macro_f1_percent']:.2f} $\\pm$ {fp[c,m]['sd_percent']:.2f}" for c in levels] for m in methods])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, ax = plt.subplots(figsize=(8.2,4.0))
for method in ['identity','gamma','clahe','retinex','diffusion']:
    y=[100*ap[v,method]['macro_f1_mean'] for v in variants]
    ax.plot(range(4),y,'o-',label=names[method],linewidth=1.7)
ax.set_xticks(range(4),['Chỉ Gamma','Gamma + Gaussian','Gamma + Poisson','Gamma + cả hai'])
ax.set(ylabel='Macro-F1 (%)',ylim=(25,103))
ax.grid(axis='y',alpha=.2); ax.legend(ncol=3,frameon=False)
fig.tight_layout();fig.savefig(FIG/'ablation.pdf');plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(9,3.5))
x=np.linspace(0,1,500)
for gamma in [1,1.5,2.2,3]:axes[0].plot(x,x**gamma,label=f'g = {gamma}')
for gamma in [1,.8,.6,.4]:axes[1].plot(x,x**gamma,label=f'a = {gamma}')
for ax,title in zip(axes,['Biến đổi làm tối','Biến đổi tăng sáng']):
    ax.set(xlabel='Cường độ đầu vào',ylabel='Cường độ đầu ra',title=title,xlim=(0,1),ylim=(0,1))
    ax.legend(frameon=False);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(FIG/'gamma_curves.pdf');plt.close(fig)

sources = {
 'notebook':{'path':str(notebook_path.relative_to(ROOT)),'sha256':hashlib.sha256(notebook_path.read_bytes()).hexdigest()},
 'fulltest':{'path':'README.md','sha256':hashlib.sha256((ROOT/'README.md').read_bytes()).hexdigest(),'status':'Reported values only; full-test raw artifacts absent locally'},
 'figures':{name:{'cell_index_zero_based':c,'output_index_zero_based':o} for name,(c,o) in images.items()},
 'subset':{'runs':len(runs),'images_per_run':430,'training_seeds':[11,22,33],'noise_seeds':[101,202,303],
           'verification':'Reaggregated notebook summaries; original notebook reports prediction-level verification. Raw predictions not available locally.'},
 'ablation':{'source':'Notebook cell 20 output 1; displayed values rounded to 4 decimals'},
 'excluded':'No legacy classifier experiment artifacts read by this script.'}
(DATA/'provenance.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
print('Extracted ResNet-18: 219 runs; 25 aggregated rows; 32 ablation rows; 24 full-test values.')
print('Clean subset Macro-F1:',points['clean','clean']['macro_f1_mean'])
