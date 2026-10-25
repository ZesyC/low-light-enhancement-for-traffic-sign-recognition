"""Build report assets exclusively from a fully audited Kaggle results archive."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import statistics as stats
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.common import digest

METHODS=['identity','gamma','he','clahe','retinex','zero_dce','retinexformer','diffusion']
NAMES=dict(zip(METHODS,['Identity','Gamma','HE','CLAHE','MSR','Zero-DCE','Retinexformer','Diffusion']))
NAMES.update({'denoise__gaussian':'Gaussian','denoise__bilateral':'Bilateral','denoise__nlm':'NLM'})
for denoiser,label in [('gaussian','Gaussian'),('bilateral','Bilateral'),('nlm','NLM')]:
    for enhancer,ename in [('gamma','Gamma'),('clahe','CLAHE')]:
        NAMES[f'pre__{denoiser}__{enhancer}']=label+' → '+ename
        NAMES[f'post__{denoiser}__{enhancer}']=ename+' → '+label
LEVELS=['mild','medium','severe']
LEVEL_NAMES={'mild':'Nhẹ','medium':'Vừa','severe':'Nặng'}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args()
    audit=json.loads((args.evidence/'audit.json').read_text())
    if audit['archive_sha256']!=digest(args.archive) or set(audit['scopes'])!={'baseline_subset','baseline_fulltest','extension_subset','extension_fulltest'}:
        raise ValueError('Use verified_results --require-extension before building report assets')
    out=ROOT/'report/latex';fig=out/'images';tables=out/'tables';data=ROOT/'report/evidence/final'
    for folder in [fig,tables,data]:folder.mkdir(parents=True,exist_ok=True)
    shutil.copy2(args.evidence/'audit.json',data/'audit.json')
    points={}
    for scope in audit['scopes']:
        points[scope]={}
        for row in read_csv(args.evidence/scope/'aggregate.csv'):
            points[scope][row['condition'],row['method']]={k:float(v) for k,v in row.items() if k not in ['condition','method']}
        for name in ['aggregate.csv','runs.csv','correlations.csv']:
            shutil.copy2(args.evidence/scope/name,data/(scope+'_'+name))
    b=points['baseline_subset'];f=points['baseline_fulltest'];e=points['extension_fulltest'];s=points['extension_subset']
    def tex(name,rows):
        (tables/(name+'.tex')).write_text('\n'.join(' & '.join(str(v) for v in row)+r' \\' for row in rows)+'\n',encoding='utf-8')
    def label(method):return NAMES[method].replace(' → ',r' $\to$ ')
    def cell(point,metric='macro_f1'):
        return f"{100*point[metric+'_mean']:.2f} $\\pm$ {100*point[metric+'_sd']:.2f}"
    for name,p,metric in [('subset_f1',b,'macro_f1'),('subset_acc',b,'accuracy'),('fulltest',f,'macro_f1')]:
        tex(name,[[NAMES[m]]+[cell(p[c,m],metric) for c in LEVELS] for m in METHODS])
    tex('quality',[[NAMES[m]]+[f"{b[c,m]['psnr_mean_mean']:.2f} / {b[c,m]['ssim_mean_mean']:.4f}" for c in LEVELS] for m in METHODS])
    tex('latency',[[NAMES[m],f"{f['medium',m]['latency_ms_mean']:.3f}"] for m in METHODS])
    with zipfile.ZipFile(args.archive) as z:
        def rows(name):return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'),newline='')))
        selection=json.loads(z.read('08_processing_extension/validation/selection.json'))
        baseline=json.loads(z.read('04_protocol/validation/selection.json'))
        chosen=selection['chosen_method'];fullmethods=selection['fulltest_methods']
        ordered=['identity','gamma','clahe','diffusion']+[m for m in fullmethods if m not in {'identity','gamma','clahe','diffusion'}]
        tex('extension_fulltest',[[label(m)]+[cell(e[c,m]) for c in LEVELS] for m in ordered])
        tex('extension_acc',[[label(m)]+[cell(e[c,m],'accuracy') for c in LEVELS] for m in ordered])
        tex('extension_quality',[[label(m)]+[f"{e[c,m]['psnr_mean_mean']:.2f} / {e[c,m]['ssim_mean_mean']:.4f}" for c in LEVELS] for m in ordered])
        extmethods=[r['method'] for r in selection['scores'] if r['category']!='control']
        tex('extension_subset',[[label(m)]+[cell(s[c,m]) for c in LEVELS] for m in extmethods])
        tex('extension_validation',[[label(r['method']),f"{100*r['macro_f1']:.2f}"] for r in sorted(selection['scores'],key=lambda r:-r['macro_f1'])])
        write_csv(data/'validation_scores.csv',[{'method':r['method'],'category':r['category'],'macro_f1':r['macro_f1'],'latency_ms':r['latency_ms']} for r in selection['scores']])
        for name,prefix in [('transitions','08_processing_extension/fulltest/analysis/'),('per_class_aggregate','08_processing_extension/fulltest/analysis/'),('order_comparisons','08_processing_extension/fulltest/analysis/')]:
            (data/(name+'.csv')).write_bytes(z.read(prefix+name+'.csv'))
        transitions=rows('08_processing_extension/fulltest/analysis/transitions.csv')
        transition_summary={}
        for m in ['clahe','diffusion',chosen]:
            for c in LEVELS:
                part=[r for r in transitions if r['method']==m and r['condition']==c]
                transition_summary[c,m]={k:stats.mean(float(r[k]) for r in part) for k in ['helped','harmed','net_correct']}
        tex('paired_transitions',[[label(m),LEVEL_NAMES[c],f"{transition_summary[c,m]['helped']:.2f}",f"{transition_summary[c,m]['harmed']:.2f}",f"{100*transition_summary[c,m]['net_correct']/12630:.2f}"] for m in ['clahe','diffusion',chosen] for c in LEVELS])
        classes=rows('08_processing_extension/fulltest/analysis/per_class_aggregate.csv')
        hard=sorted([r for r in classes if r['condition']=='severe' and r['method']==chosen],key=lambda r:float(r['f1_mean']))[:5]
        tex('hard_classes',[[r['class_id'],f"{100*float(r['f1_mean']):.2f}",f"{100*float(r['f1_sd']):.2f}",f"{float(r['delta_f1_pp']):.2f}"] for r in hard])
        plot_methods=['identity','clahe','diffusion',chosen]
        protocol=json.loads(z.read('08_processing_extension/fulltest/protocol.json'))
        summaries=rows('08_processing_extension/fulltest/summary.csv')
        with np.load(io.BytesIO(z.read('08_processing_extension/fulltest/confusion_matrices.npz')),allow_pickle=False) as npz:
            matrices=dict(zip(npz['run_ids'],npz['matrices']))
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        figure,axes=plt.subplots(2,2,figsize=(10,9))
        pairs=[]
        for ax,m in zip(axes.flat,plot_methods):
            run=next(r for r in summaries if r['condition']=='severe' and r['method']==m and r['train_seed']=='11' and r['noise_seed']=='101')
            matrix=matrices[run['run_id']].astype(float)
            normalized=np.divide(matrix,matrix.sum(1,keepdims=True),out=np.zeros_like(matrix),where=matrix.sum(1,keepdims=True)!=0)
            image=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
            ax.set(title=NAMES[m],xlabel='Lớp dự đoán',ylabel='Lớp thật',xticks=range(0,43,6),yticks=range(0,43,6))
            counts=sum((matrices[r['run_id']] for r in summaries if r['condition']=='severe' and r['method']==m))
            rates=np.divide(counts,counts.sum(1,keepdims=True),out=np.zeros_like(counts,dtype=float),where=counts.sum(1,keepdims=True)!=0)
            np.fill_diagonal(rates,0)
            for index in np.argsort(rates.ravel())[-3:][::-1]:
                a,p=divmod(int(index),43);pairs.append([label(m),a,p,f'{100*rates[a,p]:.2f}'])
        figure.subplots_adjust(wspace=.30,hspace=.3,right=.84)
        color_axis=figure.add_axes([.88,.18,.025,.64])
        figure.colorbar(image,cax=color_axis,label='Tỷ lệ trong lớp thật')
        figure.savefig(fig/'paired_confusions.png',dpi=180,bbox_inches='tight');plt.close(figure)
        tex('confusion_pairs',pairs)
        figure,ax=plt.subplots(figsize=(10,3.5))
        for m in plot_methods:
            part=sorted([r for r in classes if r['condition']=='severe' and r['method']==m],key=lambda r:int(r['class_id']))
            ax.plot([int(r['class_id']) for r in part],[100*float(r['f1_mean']) for r in part],marker='.',label=NAMES[m],linewidth=1)
        ax.set(xlabel='Class ID',ylabel='F1 (%)',xticks=range(0,43,2),ylim=(0,103));ax.legend(ncol=2,frameon=False,fontsize=9);ax.grid(alpha=.2)
        figure.tight_layout();figure.savefig(fig/'per_class_severe.png',dpi=180);plt.close(figure)
        # Actual paired examples: one fixed checkpoint/noise/condition; no historical model labels.
        paired={m:{} for m in ['identity',chosen]}
        with z.open('08_processing_extension/fulltest/predictions.csv') as binary:
            for r in csv.DictReader(io.TextIOWrapper(binary,encoding='utf-8',newline='')):
                if r['condition']=='medium' and r['train_seed']=='11' and r['noise_seed']=='101' and r['method'] in paired:
                    paired[r['method']][r['sample_id']]=r
        cases=[]
        for category in ['helped','harmed','still_wrong']:
            for sid in sorted(paired[chosen]):
                a,before=paired[chosen][sid],paired['identity'][sid]
                old=before['prediction']==before['class_id'];new=a['prediction']==a['class_id']
                group='helped' if new and not old else 'harmed' if old and not new else 'still_wrong' if not new else 'still_correct'
                if group==category:
                    cases.append(dict(sample_id=sid,category=group,class_id=a['class_id'],before=before['prediction'],after=a['prediction']))
                    if sum(r['category']==category for r in cases)==2:break
        write_csv(data/'representative_cases.csv',cases)
        cn={'helped':'Được giúp','harmed':'Bị làm sai','still_wrong':'Vẫn sai'}
        tex('representative_cases',[[r['sample_id'].rsplit('/',1)[-1],cn[r['category']],r['class_id'],r['before'],r['after']] for r in cases])
        # Ablation is kept separate from degradation-level test metrics.
        ab=rows('07_supplemental/ablation/summary.csv');within=defaultdict(list)
        for r in ab:within[r['variant'],r['method'],r['train_seed']].append(float(r['macro_f1']))
        across=defaultdict(list)
        for (v,m,_),vals in within.items():across[v,m].append(stats.mean(vals))
        ablation={key:stats.mean(vals) for key,vals in across.items()}
        variants=['gamma_only','gamma_gaussian','gamma_poisson','gamma_both']
        tex('ablation',[[NAMES[m]]+[f'{100*ablation[v,m]:.2f}' for v in variants] for m in METHODS])
        write_csv(data/'ablation.csv',[dict(variant=v,method=m,macro_f1_mean=value) for (v,m),value in ablation.items()])
        figure,ax=plt.subplots(figsize=(8.5,3.5))
        for m in ['identity','clahe','retinex','zero_dce','diffusion']:ax.plot(range(4),[100*ablation[v,m] for v in variants],'o-',label=NAMES[m])
        ax.set(xticks=range(4),xticklabels=['Chỉ Gamma','+ Gaussian','+ Poisson','+ Cả hai'],ylabel='Macro-F1 (%)',ylim=(20,103));ax.legend(ncol=3,frameon=False);ax.grid(axis='y',alpha=.2)
        figure.tight_layout();figure.savefig(fig/'ablation.pdf');plt.close(figure)
        epochs=[]
        for seed in [11,22,33]:
            learning=rows(f'05_training/learning_{seed}.csv')
            valid=[r for r in learning if r.get('phase',r.get('split'))=='validation']
            # Checkpoint choice uses min_delta; read its recorded epoch rather than reselecting.
            import torch
            checkpoint=torch.load(io.BytesIO(z.read(f'05_training/cnn_{seed}.pt')),map_location='cpu',weights_only=True)
            epochs.append(dict(seed=seed,best_epoch=checkpoint.get('epoch'),last_epoch=max(int(r['epoch']) for r in learning)))
        facts=dict(archive_sha256=audit['archive_sha256'],chosen_method=chosen,baseline_selected=baseline['chosen_method'],
            baseline_selection_scores=[{k:r[k] for k in ['method','macro_f1']} for r in baseline['scores']],
            validation_scores=[{k:r[k] for k in ['method','macro_f1']} for r in selection['scores']],
            baseline_parameters=baseline['enhancers'],denoiser_parameters={m:selection['registry']['denoise__'+m]['steps'][0]['params'] for m in ['gaussian','bilateral','nlm']},
            checkpoints=epochs,transitions={c+'|'+m:v for (c,m),v in transition_summary.items()},
            hard_classes=hard,points={scope:{c+'|'+m:v for (c,m),v in p.items()} for scope,p in points.items()})
        (data/'summary.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
    corr=read_csv(args.evidence/'baseline_fulltest/correlations.csv')
    tex('correlations',[[LEVEL_NAMES[r['condition']],'PSNR' if r['quality']=='psnr_mean' else 'SSIM','Accuracy' if r['metric']=='accuracy' else 'Macro-F1',f"{float(r['pearson']):.3f}",f"{float(r['spearman']):.3f}"] for r in corr])
    for name,p,methods in [('subset_f1',b,METHODS),('extension_f1',e,['identity','clahe','diffusion','denoise__gaussian',chosen,'post__gaussian__clahe'])]:
        figure,ax=plt.subplots(figsize=(8.5,3.6))
        for m in methods:ax.plot(range(3),[100*p[c,m]['macro_f1_mean'] for c in LEVELS],'o-',label=NAMES[m],linewidth=1.5)
        ax.set(xticks=range(3),xticklabels=['Nhẹ','Vừa','Nặng'],ylabel='Macro-F1 (%)',ylim=(20,101));ax.legend(ncol=3,frameon=False,fontsize=9);ax.grid(alpha=.2)
        figure.tight_layout();figure.savefig(fig/(name+'.png'),dpi=180);plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(9.3,3.4))
    for ax,q in zip(axes,['psnr_mean','ssim_mean']):
        for i,m in enumerate(METHODS):
            x=f['medium',m][q+'_mean'];y=100*f['medium',m]['macro_f1_mean']
            offsets=({'identity':(-12,9),'gamma':(5,-12),'retinexformer':(5,-10),'clahe':(-32,7),'zero_dce':(-45,-12),'diffusion':(5,5)}
                     if q=='psnr_mean' else {'identity':(-12,-13),'gamma':(-20,10),'retinexformer':(5,-12),'clahe':(-25,9),'zero_dce':(-57,-12),'diffusion':(-43,6)})
            ax.scatter(x,y,s=30);ax.annotate(NAMES[m],(x,y),xytext=offsets.get(m,(4,5)),textcoords='offset points',fontsize=8)
        ax.margins(x=.15,y=.08)
        ax.set(xlabel='PSNR (dB)' if q=='psnr_mean' else 'SSIM',ylabel='Macro-F1 (%)');ax.grid(alpha=.2)
    figure.tight_layout();figure.savefig(fig/'subset_scatter.png',dpi=180);plt.close(figure)
    provenance={'source':'Kaggle notebook version 2, completed; raw predictions audited for all four scopes',
        'archive_sha256':audit['archive_sha256'],'audit':'report/evidence/final/audit.json',
        'derived_tables':'report/evidence/final/','selected_method':facts['chosen_method'],
        'illustrative_images':'degradation, classical_default, parameters, enhancers_selected: historical image-processing illustrations only, never current classifier evidence',
        'builder_sha256':digest(__file__),'real_lowlight_evaluated':False}
    (ROOT/'report/evidence/provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Built verified report assets:',facts['chosen_method'],'| source',audit['archive_sha256'])


if __name__=='__main__':main()
