"""Validation-first study of denoising, enhancement and operation order."""
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from src.common import (config_id,digest,read_csv,write_csv,write_json,read_image,synchronize)
from src.data import subset_rows
from src.degradation import degrade
from src.experiment import load_cnn,manifest_rows,source_code_hash
from src.enhancement import provenance
from src.metrics import classification,image_metrics
from src.model import image_tensor
from .processing_pipeline import Pipeline,pipeline_provenance


def study_hash():
    return config_id({p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('processing_pipeline.py')]})


def _enhance(method,params):
    return {'kind':'enhance','method':method,'params':deepcopy(params)}


def _denoise(method,params):
    return {'kind':'denoise','method':method,'params':deepcopy(params)}


def controls(baseline):
    result={'identity':{'steps':[]}}
    for method in dict.fromkeys(['gamma','clahe',baseline['chosen_method']]):
        if method!='identity':result[method]={'steps':[_enhance(method,baseline['enhancers'][method])]}
    return result


def choose_fulltest_methods(scores):
    """Select matched orders, their standalone controls and winners using validation only."""
    points={r['method']:r for r in scores}
    base=sorted(r['method'] for r in scores if r['category']=='control')
    singles=sorted((r for r in scores if r['category']=='denoise'),key=lambda r:(-r['macro_f1'],r['method']))
    pairs=[]
    for r in scores:
        if r['category']=='denoise_then_enhance':
            mate='post__'+r['method'].removeprefix('pre__')
            pairs.append(((r['macro_f1']+points[mate]['macro_f1'])/2,r['method'],mate,r['denoiser']))
    if not pairs or not singles:raise ValueError('Missing controlled order pairs or denoisers')
    best=sorted(pairs,key=lambda x:(-x[0],x[1]))[0]
    overall=sorted(scores,key=lambda r:(-r['macro_f1'],r['latency_ms'],r['method']))[0]['method']
    return list(dict.fromkeys(base+[singles[0]['method'],'denoise__'+best[3],best[1],best[2],overall]))


def _reference_images(root,rows):
    """Decode each clean PNG once per scope; keep read-only float32 RGB in RAM."""
    references={}
    for row in rows:
        image=read_image(Path(root)/row['crop_path']);image.setflags(write=False)
        references[row['sample_id']]=image
    print('Clean RGB references in RAM:',len(references),'images,',
          round(sum(a.nbytes for a in references.values())/1024**2,1),'MiB',flush=True)
    return references


def _process_group(root,rows,config,condition,noise,spec,models,device,references=None):
    """Stream enhanced RGB batches to all three CNNs without intermediate image files."""
    def reference(row):
        return references[row['sample_id']] if references is not None else read_image(Path(root)/row['crop_path'])
    pipeline=Pipeline(spec,device)
    if condition!='clean':
        first=rows[0]
        warm=degrade(reference(first),**config['conditions'][condition],
                     noise_seed=int(noise),sample_id=first['sample_id'],condition=condition)
        for _ in range(2):pipeline(warm,first['sample_id'])
        synchronize(device)
    predictions={saved['seed']:[] for _,saved in models}
    batch_size=config['cnn']['batch_size']
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]
        tensors,records=[],[]
        for row in batch:
            clean=reference(row)
            dark=clean if condition=='clean' else degrade(clean,**config['conditions'][condition],
                noise_seed=int(noise),sample_id=row['sample_id'],condition=condition)
            synchronize(device);began=time.perf_counter()
            enhanced=clean if condition=='clean' else pipeline(dark,row['sample_id'])
            synchronize(device)
            latency=0.0 if condition=='clean' else 1000*(time.perf_counter()-began)
            tensors.append(image_tensor(enhanced))
            records.append({'sample_id':row['sample_id'],'group_id':row['group_id'],
                'source_id':row['source_id'],'class_id':int(row['class_id']),'noise_seed':noise,
                **image_metrics(clean,enhanced,config['ssim']),'latency_ms':latency,'output_path':''})
        images=torch.stack(tensors).to(device)
        with torch.inference_mode():
            for model,saved in models:
                probabilities=model(images).softmax(1).cpu().numpy()
                if not np.isfinite(probabilities).all():raise ValueError('Nonfinite classifier probabilities')
                for record,p in zip(records,probabilities):
                    predictions[saved['seed']].append({**record,'prediction':int(p.argmax()),
                        'true_probability':float(p[record['class_id']])})
    return predictions


def _score(rows):
    return classification([int(r['class_id']) for r in rows],[int(r['prediction']) for r in rows])


def tune(root,baseline_path,checkpoint_paths,extension_config,device,output):
    root,output=Path(root),Path(output)
    baseline=json.loads(Path(baseline_path).read_text())
    plan=json.loads(Path(extension_config).read_text())
    all_rows,manifest_hash=manifest_rows(root)
    if baseline['manifest_sha256']!=manifest_hash or baseline['code_hash']!=source_code_hash():
        raise ValueError('Baseline selection is stale; reuse CNNs but revalidate the baseline first')
    if baseline.get('pilot',True) or baseline.get('status')!='completed':raise ValueError('Need completed baseline validation')
    if {digest(p) for p in checkpoint_paths}!=set(baseline['checkpoints'].values()):raise ValueError('Checkpoint mismatch')
    for method,params in baseline['enhancers'].items():
        if provenance(method,params)!=baseline['enhancer_provenance'][method]:raise ValueError('Baseline enhancer changed')
    lock={'study_hash':study_hash(),'baseline_selection_sha256':digest(baseline_path),
          'plan_sha256':digest(extension_config),'manifest_sha256':manifest_hash,'code_hash':source_code_hash()}
    if (output/'validation_lock.json').exists() and json.loads((output/'validation_lock.json').read_text())!=lock:
        raise ValueError('Validation output belongs to other inputs/code')
    write_json(output/'validation_lock.json',lock)
    if (output/'selection.json').exists():
        previous=json.loads((output/'selection.json').read_text())
        if previous['study_hash']==study_hash() and previous['baseline_selection_sha256']==digest(baseline_path) and previous['plan_sha256']==digest(extension_config):
            return previous
        raise ValueError('Extension selection belongs to different inputs/code; use a new output')
    rows=subset_rows(all_rows,'validation',plan['validation_per_class'])
    if {r['sample_id'] for r in rows}!=set(baseline['sample_ids']):raise ValueError('Validation subset differs from baseline')
    models=[load_cnn(p,manifest_hash,device,baseline['config']['cnn']) for p in checkpoint_paths]
    if {s['seed'] for _,s in models}!={11,22,33}:raise ValueError('Require classifier seeds 11/22/33')
    cfg=deepcopy(baseline['config']);cfg['enhancers'].update(baseline['enhancers'])
    references=_reference_images(root,rows)
    registry=controls(baseline);scores=[];details=[]
    def measure(name,spec,category,denoiser=''):
        values=[];latencies=[]
        candidate_id=config_id(spec)
        for condition in cfg['conditions']:
            for noise in cfg['noise_seeds']:
                planned={s['seed']:{'method':name,'candidate_id':candidate_id,'condition':condition,'noise_seed':noise,'train_seed':s['seed']} for _,s in models}
                predictions={}
                for seed,run in planned.items():
                    rid=config_id(run);record=output/'completed'/f'{rid}.json';path=output/'predictions'/f'{rid}.csv'
                    if record.exists():
                        if not path.exists() or digest(path)!=json.loads(record.read_text())['prediction_sha256']:
                            raise ValueError('Completed validation predictions changed')
                        predictions[seed]=_verified_predictions(path,rows,run)
                if len(predictions)!=len(models):
                    missing=[m for m in models if m[1]['seed'] not in predictions]
                    predictions.update(_process_group(root,rows,cfg,condition,noise,spec,missing,device,references))
                latencies.append(np.mean([float(r['latency_ms']) for r in next(iter(predictions.values()))]))
                for seed,preds in predictions.items():
                    score=_score(preds)
                    run={'method':name,'candidate_id':candidate_id,'condition':condition,'noise_seed':noise,'train_seed':seed}
                    rid=config_id(run)
                    write_csv(output/'predictions'/f'{rid}.csv',preds)
                    write_json(output/'completed'/f'{rid}.json',{'prediction_sha256':digest(output/'predictions'/f'{rid}.csv')})
                    details.append({**run,'run_id':rid,'n_images':len(rows),**{k:score[k] for k in ['accuracy','macro_f1','weighted_f1']}})
                    values.append(score['macro_f1'])
                write_csv(output/'summary.csv',details)
        result={'method':name,'category':category,'denoiser':denoiser,'candidate_id':candidate_id,
                'macro_f1':float(np.mean(values)),'latency_ms':float(np.mean(latencies)),'spec':spec,'origin':'extension_validation'}
        print('Extension validation:',name,candidate_id,result['macro_f1'],flush=True)
        return result
    baseline_points={r['method']:r for r in baseline['scores']}
    for name,spec in registry.items():
        r=baseline_points[name]
        scores.append({'method':name,'category':'control','denoiser':'','macro_f1':r['macro_f1'],
                       'latency_ms':r['latency_ms'],'spec':spec,'origin':'same_frozen_baseline_validation'})
    chosen_strengths={}
    for method,parameters in plan['denoisers'].items():
        candidates=[measure('denoise__'+method,{'steps':[_denoise(method,p)]},'denoise',method) for p in parameters]
        selected=sorted(candidates,key=lambda r:(-r['macro_f1'],r['candidate_id']))[0]
        scores.append(selected);registry[selected['method']]=selected['spec']
        chosen_strengths[method]=selected['spec']['steps'][0]['params']
        write_json(output/f'strength_{method}.json',candidates)
    # Fix one strength for each denoiser before comparing the two orders.
    for denoiser,parameters in chosen_strengths.items():
        for enhancer in plan['composition_enhancers']:
            d=_denoise(denoiser,parameters);e=_enhance(enhancer,baseline['enhancers'][enhancer])
            for prefix,steps,category in [('pre',[d,e],'denoise_then_enhance'),('post',[e,d],'enhance_then_denoise')]:
                name=f'{prefix}__{denoiser}__{enhancer}';spec={'steps':steps}
                registry[name]=spec;scores.append(measure(name,spec,category,denoiser))
    full_methods=choose_fulltest_methods(scores)
    chosen=sorted(scores,key=lambda r:(-r['macro_f1'],r['latency_ms'],r['method']))[0]['method']
    selected={'study':'denoise_order_extension','status':'completed','pilot':False,'study_hash':study_hash(),
        'code_hash':source_code_hash(),'baseline_selection_sha256':digest(baseline_path),'plan_sha256':digest(extension_config),
        'manifest_sha256':manifest_hash,'sample_ids':[r['sample_id'] for r in rows],'config':cfg,
        'checkpoints':{str(Path(p).resolve()):digest(p) for p in checkpoint_paths},'registry':registry,'scores':scores,
        'chosen_method':chosen,'fulltest_methods':full_methods,'methods':list(registry),'plan':plan,
        'pipeline_provenance':{name:pipeline_provenance(spec) for name,spec in registry.items()},
        'selection_rule':'Fixed strengths for order comparisons; paired mean validation score selects full-test order pair; no test selection.'}
    write_json(output/'selection.json',selected)
    return selected


def freeze(root,selection_path,output,per_class=10):
    root,output=Path(root),Path(output)
    selection=json.loads(Path(selection_path).read_text())
    all_rows,manifest_hash=manifest_rows(root)
    if selection['study_hash']!=study_hash() or selection['code_hash']!=source_code_hash() or selection['manifest_sha256']!=manifest_hash:
        raise ValueError('Extension selection/code/data changed')
    if selection.get('pilot',True) or selection['status']!='completed':raise ValueError('Incomplete/pilot selection')
    rows=subset_rows(all_rows,'test',per_class,42)
    methods=selection['methods'] if per_class is not None else selection['fulltest_methods']
    checkpoints=[]
    for path,sha in selection['checkpoints'].items():
        if digest(path)!=sha:raise ValueError('Checkpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        checkpoints.append({'path':path,'sha256':sha,'seed':saved['seed']})
    if {p['seed'] for p in checkpoints}!={11,22,33}:raise ValueError('Missing classifier seed')
    runs=[]
    for c in checkpoints:
        runs.append({'condition':'clean','noise_seed':'','method':'clean','train_seed':c['seed']})
        for condition in selection['config']['conditions']:
            for noise in selection['config']['noise_seeds']:
                for method in methods:runs.append({'condition':condition,'noise_seed':noise,'method':method,'train_seed':c['seed']})
    protocol={'study':'denoise_order_extension','pilot':False,'selection_sha256':digest(selection_path),
        'selection':{**selection,'methods':methods},'sample_ids':[r['sample_id'] for r in rows],
        'subset_id':'official_test' if per_class is None else config_id([r['sample_id'] for r in rows]),
        'manifest_sha256':manifest_hash,'config':selection['config'],'study_hash':study_hash(),
        'code_hash':source_code_hash(),'checkpoints':checkpoints,'runs':runs,
        'registry':{name:selection['registry'][name] for name in methods},
        'pipeline_provenance':{name:selection['pipeline_provenance'][name] for name in methods}}
    if output.exists():
        if json.loads(output.read_text())!=protocol:raise ValueError('Frozen extension protocol differs')
        return protocol
    write_json(output,protocol)
    return protocol


def _verified_predictions(path,rows,run):
    predictions=read_csv(path)
    by_id={r['sample_id']:r for r in rows}
    if len(predictions)!=len(rows) or {r['sample_id'] for r in predictions}!=set(by_id):raise ValueError('Prediction coverage mismatch')
    for p in predictions:
        truth=by_id[p['sample_id']]
        if int(p['class_id'])!=int(truth['class_id']) or p['group_id']!=truth['group_id'] or str(p['noise_seed'])!=str(run['noise_seed']):
            raise ValueError('Predictions differ from current ground truth/groups/noise')
    return predictions


def evaluate(root,protocol_path,output,device,baseline_output=None):
    from .research_evidence import audited_runs
    root,output=Path(root),Path(output)
    protocol=json.loads(Path(protocol_path).read_text())
    all_rows,manifest_hash=manifest_rows(root)
    if protocol['study_hash']!=study_hash() or protocol['code_hash']!=source_code_hash() or manifest_hash!=protocol['manifest_sha256']:
        raise ValueError('Frozen extension code/data changed')
    by_id={r['sample_id']:r for r in all_rows if r['split']=='test'}
    if len(set(protocol['sample_ids']))!=len(protocol['sample_ids']) or not set(protocol['sample_ids'])<=by_id.keys():raise ValueError('Invalid test IDs')
    rows=[by_id[i] for i in protocol['sample_ids']]
    for c in protocol['checkpoints']:
        if digest(c['path'])!=c['sha256']:raise ValueError('Checkpoint changed')
    for method,spec in protocol['registry'].items():
        if pipeline_provenance(spec)!=protocol['pipeline_provenance'][method]:raise ValueError('Pipeline changed')
    lock=output/'protocol.json'
    if lock.exists() and json.loads(lock.read_text())!=protocol:raise ValueError('Output belongs to another protocol')
    write_json(lock,protocol)
    old_summaries={r['run_id']:r for r in read_csv(output/'summary.csv')} if (output/'summary.csv').exists() else {}
    old_states={config_id({k:s[k] for k in ['condition','method','noise_seed','train_seed']}):s for s in json.loads((output/'runs.json').read_text())} if (output/'runs.json').exists() else {}
    states={config_id(r):dict(r,status='planned') for r in protocol['runs']};summaries={}
    def keep(run,predictions,origin):
        rid=config_id(run);score=_score(predictions)
        write_csv(output/'predictions'/f'{rid}.csv',predictions)
        write_json(output/'classes'/f'{rid}.json',score)
        summaries[rid]={'dataset':'gtsrb','split':'test','protocol':'processing_extension','subset_id':protocol['subset_id'],
            **run,'run_id':rid,'config_id':config_id(protocol['registry'].get(run['method'],{'steps':[]})),
            'n_images':len(rows),**{k:score[k] for k in ['accuracy','macro_f1','weighted_f1']},
            'psnr_mean':float(np.mean([float(r['psnr']) for r in predictions])),
            'ssim_mean':float(np.mean([float(r['ssim']) for r in predictions])),
            'latency_ms':float(np.mean([float(r['latency_ms']) for r in predictions]))}
        states[rid]=dict(run,status='completed',origin=origin,prediction_sha256=digest(output/'predictions'/f'{rid}.csv'))
        write_csv(output/'summary.csv',list(summaries.values()));write_json(output/'runs.json',list(states.values()))
    for run in protocol['runs']:
        rid=config_id(run);path=output/'predictions'/f'{rid}.csv';state=old_states.get(rid,{})
        if state.get('status')=='completed' and rid in old_summaries and path.is_file():
            if digest(path)!=state['prediction_sha256']:raise ValueError('Completed prediction file changed')
            preds=_verified_predictions(path,rows,run);score=_score(preds)
            if any(not math.isclose(score[k],float(old_summaries[rid][k]),abs_tol=1e-8,rel_tol=0) for k in ['accuracy','macro_f1','weighted_f1']):raise ValueError('Completed metric changed')
            summaries[rid]=old_summaries[rid];states[rid]=state
    reused=[]
    if baseline_output:
        baseline,runs=audited_runs(baseline_output)
        same=(baseline['manifest_sha256']==manifest_hash and baseline['code_hash']==protocol['code_hash']
            and baseline['config']==protocol['config'] and set(baseline['sample_ids'])==set(protocol['sample_ids'])
            and {c['seed']:c['sha256'] for c in baseline['checkpoints']}=={c['seed']:c['sha256'] for c in protocol['checkpoints']})
        if not same:raise ValueError('Baseline reuse requires identical data/config/checkpoints/implementation')
        lookup={r['run_id']:(r,p) for r,p,_ in runs}
        for run in protocol['runs']:
            rid=config_id(run)
            if rid in summaries or rid not in lookup:continue
            # Only baseline controls have the same 4-field run ID and same locked enhancer parameters.
            if run['method']!='clean':
                expected={'steps':[]} if run['method']=='identity' else {'steps':[_enhance(run['method'],baseline['config']['enhancers'][run['method']])]}
                if protocol['registry'][run['method']]!=expected:raise ValueError('Baseline control parameters differ')
            _,preds=lookup[rid]
            _verified_predictions(Path(baseline_output)/'predictions'/f'{rid}.csv',rows,run)
            keep(run,preds,'verified_baseline_reuse');reused.append(rid)
    groups=defaultdict(list)
    for run in protocol['runs']:
        if config_id(run) not in summaries:groups[run['condition'],str(run['noise_seed']),run['method']].append(run)
    models=[]
    if groups:models=[load_cnn(c['path'],manifest_hash,device,protocol['config']['cnn']) for c in protocol['checkpoints']]
    references=_reference_images(root,rows) if groups else None
    for (condition,noise,method),runs in groups.items():
        needed={r['train_seed'] for r in runs};active=[m for m in models if m[1]['seed'] in needed]
        spec={'steps':[]} if method=='clean' else protocol['registry'][method]
        predictions=_process_group(root,rows,protocol['config'],condition,noise,spec,active,device,references)
        for run in runs:keep(run,predictions[run['train_seed']],'extension_inference')
        print('Extension test completed:',len(summaries),'/',len(protocol['runs']),method,condition,noise,flush=True)
    write_csv(output/'summary.csv',[summaries[config_id(r)] for r in protocol['runs']])
    write_json(output/'runs.json',[states[config_id(r)] for r in protocol['runs']])
    write_json(output/'reuse.json',{'new_training':False,
        'reused_run_ids':[rid for rid,s in states.items() if s.get('origin')=='verified_baseline_reuse'],
        'source':str(baseline_output) if baseline_output else None,
        'latency_note':'Reused control latency was measured in the baseline session; cross-session timing is descriptive only.'})
    audited_runs(output)
    return output


def analyze(folder,output):
    """Standard error analysis plus matched comparison of the two orders."""
    from .research_evidence import analyze as error_analysis,audited_runs
    from src.metrics import paired_rows
    error_analysis(folder,output)
    protocol,runs=audited_runs(folder)
    index={(r['condition'],r['method'],r['train_seed'],r['noise_seed']):(r,p,s) for r,p,s in runs}
    comparisons=[]
    for before,preds,score in runs:
        if not before['method'].startswith('pre__'):continue
        mate='post__'+before['method'].removeprefix('pre__')
        key=(before['condition'],mate,before['train_seed'],before['noise_seed'])
        if key not in index:continue  # The global winner can be retained outside the selected matched pair.
        after,other,other_score=index[key]
        helped=harmed=0
        for a,b in paired_rows(preds,other):
            correct_a=a['class_id']==a['prediction'];correct_b=b['class_id']==b['prediction']
            helped+=int(not correct_a and correct_b);harmed+=int(correct_a and not correct_b)
        _,denoiser,enhancer=before['method'].split('__')
        comparisons.append({'condition':before['condition'],'denoiser':denoiser,'enhancer':enhancer,
            'train_seed':before['train_seed'],'noise_seed':before['noise_seed'],
            'pre_method':before['method'],'post_method':mate,'n_images':len(preds),
            'post_minus_pre_accuracy_pp':100*(other_score['accuracy']-score['accuracy']),
            'post_minus_pre_macro_f1_pp':100*(other_score['macro_f1']-score['macro_f1']),
            'post_helped':helped,'post_harmed':harmed,
            'post_minus_pre_psnr':float(after['psnr_mean'])-float(before['psnr_mean']),
            'post_minus_pre_ssim':float(after['ssim_mean'])-float(before['ssim_mean'])})
    if comparisons:write_csv(Path(output)/'order_comparisons.csv',comparisons)
    return output
