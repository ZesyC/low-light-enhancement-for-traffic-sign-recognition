"""Build LaTeX assets only from audited raw subset predictions supplied by the user."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from src.common import digest
from .research_evidence import audited_runs, read_csv, write_csv


def build(folder, analysis, archive, root):
    protocol, runs = audited_runs(folder)
    if len(protocol['sample_ids']) != 430 or any(s['support'] != [10]*43 for _,_,s in runs):
        raise ValueError('This report template requires the balanced 430-image subset, support 10/class/run')
    root=Path(root)
    evidence=root/'report/evidence/error_analysis'
    evidence.mkdir(parents=True,exist_ok=True)
    tables=root/'report/latex/tables'
    images=root/'report/latex/images'
    for name in ['per_class_aggregate.csv','transitions.csv','confusion_pairs.csv','examples.csv','audit.json']:
        shutil.copyfile(Path(analysis)/name,evidence/name)
    def table(name,rows):
        (tables/name).write_text('\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n',encoding='utf-8')
    transitions=read_csv(evidence/'transitions.csv')
    conditions={'mild':'Nhẹ','medium':'Vừa','severe':'Nặng'}
    tex=[]
    for method in ['clahe','diffusion']:
        for c,label in conditions.items():
            rows=[r for r in transitions if r['method']==method and r['condition']==c]
            helped=np.mean([int(r['helped']) for r in rows])
            harmed=np.mean([int(r['harmed']) for r in rows])
            tex.append(['CLAHE' if method=='clahe' else 'Diffusion',label,f'{helped:.2f}',f'{harmed:.2f}',f'{100*(helped-harmed)/len(protocol["sample_ids"]):+.2f}'])
    table('paired_transitions.tex',tex)
    per=read_csv(evidence/'per_class_aggregate.csv')
    worst=sorted((r for r in per if r['method']=='diffusion' and r['condition']=='severe'),key=lambda r:float(r['f1_mean']))[:5]
    table('hard_classes.tex',[[r['class_id'],f"{float(r['f1_mean'])*100:.2f}",f"{float(r['f1_sd'])*100:.2f}",f"{float(r['delta_f1_pp']):+.2f}"] for r in worst])
    pairs=read_csv(evidence/'confusion_pairs.csv')
    tex=[]
    for method in ['identity','clahe','diffusion']:
        groups=defaultdict(int)
        for r in pairs:
            if r['method']==method and r['condition']=='severe':
                groups[int(r['true_class']),int(r['predicted_class'])]+=int(r['count'])
        for (truth,pred),count in sorted(groups.items(),key=lambda item:(-item[1],item[0]))[:3]:
            # 9 runs, each with support 10 per true class. Descriptive average, not 90 independent images.
            tex.append([method,truth,pred,f'{100*count/90:.2f}'])
    table('confusion_pairs.tex',tex)
    cases=read_csv(Path(analysis)/'cases.csv')
    samples=[]
    for category in ['helped','harmed','still_wrong']:
        rows=sorted((r for r in cases if r['method']=='diffusion' and r['condition']=='medium' and r['train_seed']=='11' and r['noise_seed']=='101' and r['category']==category),key=lambda r:r['sample_id'])
        for r in rows[:2]:
            samples.append(r)
    write_csv(evidence/'representative_cases.csv',samples)
    names={'helped':'Được giúp','harmed':'Bị làm sai','still_wrong':'Vẫn sai'}
    table('representative_cases.tex',[[r['sample_id'].split('/')[-1],names[r['category']],r['class_id'],r['before'],r['after']] for r in samples])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':16})
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,method in zip(axes,['identity','clahe','diffusion']):
        _,_,score=next((r,p,s) for r,p,s in runs if r['method']==method and r['condition']=='severe' and r['train_seed']=='11' and r['noise_seed']=='101')
        matrix=np.asarray(score['confusion_matrix'],dtype=float)
        matrix=matrix/matrix.sum(1,keepdims=True)
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
        ax.set(title=method,xlabel='Predicted class',ylabel='True class',xticks=[0,10,20,30,40],yticks=[0,10,20,30,40])
    fig.colorbar(im,ax=axes,label='Fraction within true class',shrink=0.8)
    fig.savefig(images/'paired_confusions.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
    for method in ['identity','clahe','diffusion']:
        rows=sorted((r for r in per if r['method']==method and r['condition']=='severe'),key=lambda r:int(r['class_id']))
        ax.plot([int(r['class_id']) for r in rows],[100*float(r['f1_mean']) for r in rows],'.-',label=method,linewidth=1)
    ax.set(xlabel='GTSRB class ID',ylabel='Mean per-class F1 (%)',ylim=(0,105),xticks=list(range(0,43,2)))
    ax.grid(axis='y',alpha=.2);ax.legend(ncol=3)
    fig.savefig(images/'per_class_severe.png',dpi=180);plt.close(fig)
    hashes={str(p.relative_to(root)):digest(p) for p in sorted(evidence.iterdir()) if p.is_file()}
    checkpoint_checks=[]
    for c in protocol['checkpoints']:
        p=Path(folder).parent/'train'/Path(c['path']).name
        matched=p.exists() and digest(p)==c['sha256']
        if not matched:
            raise ValueError('Restored checkpoint hash mismatch')
        checkpoint_checks.append({'seed':c['seed'],'sha256':c['sha256'],'verified':True})
    metadata={'archive_name':Path(archive).name,'archive_sha256':digest(archive),
              'scope':'subset 430 images, 10 per class; not full-test','runs':len(runs),
              'predictions_recomputed':sum(len(p) for _,p,_ in runs),'checkpoint_hash_checks':checkpoint_checks,
              'ground_truth_check':'Internal agreement across all prediction runs. Official image/annotation files not re-downloaded for independent label verification.',
              'files':hashes,'fulltest':'Raw predictions absent; reported CSV retained separately.',
              'real_lowlight':'Not yet collected or evaluated.'}
    (evidence/'source.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('.'))
    a=p.parse_args()
    print(build(a.input,a.analysis,a.archive,a.root))
