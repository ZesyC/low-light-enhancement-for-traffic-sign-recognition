"""Fresh Kaggle execution with official inputs, bounded caches and separate GPU workers."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys


def parallel_evaluate(root, protocol_path, output, devices):
    from src.common import config_id, read_csv, write_csv, write_json
    from .research_evidence import audited_runs
    root,output=Path(root),Path(output)
    master=json.loads(Path(protocol_path).read_text())
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'protocol.json'
    if lock.exists() and json.loads(lock.read_text()) != master:
        raise ValueError('Output belongs to another experiment')
    write_json(lock,master)
    groups=defaultdict(list)
    for r in master['runs']:groups[r['condition'],str(r['noise_seed']),r['method']].append(r)
    assignments=[[] for _ in devices]
    loads=[0 for _ in devices]
    costs={'diffusion':30,'retinexformer':4,'zero_dce':2}
    for key,runs in sorted(groups.items(),key=lambda item:(-costs.get(item[0][2],1),item[0])):
        rank=min(range(len(devices)),key=lambda i:loads[i])
        assignments[rank].extend(runs);loads[rank]+=costs.get(key[2],1)
    workers=[]
    for rank,(device,runs) in enumerate(zip(devices,assignments)):
        if not runs:continue
        partial=deepcopy(master);partial['runs']=runs;partial['pilot']=True
        protocol=output/'_workers'/f'worker_{rank}.json'
        write_json(protocol,partial)
        dest=output/'_workers'/f'worker_{rank}'
        command=[sys.executable,'-m','src.tools.fresh_run','evaluate-worker','--root',str(root),
                 '--protocol',str(protocol),'--output',str(dest),'--device',device]
        workers.append((dest,subprocess.Popen(command,cwd=root)))
    codes=[process.wait() for _,process in workers]
    if any(codes):raise RuntimeError(f'Evaluation worker exit codes: {codes}')
    summaries=[]
    for folder,_ in workers:
        audited_runs(folder)
        summaries.extend(read_csv(folder/'summary.csv'))
        for name in ['predictions','classes']:
            (output/name).mkdir(exist_ok=True)
            for p in (folder/name).glob('*'):shutil.copy2(p,output/name/p.name)
    by_id={r['run_id']:r for r in summaries}
    ordered=[by_id[config_id(r)] for r in master['runs']]
    write_csv(output/'summary.csv',ordered)
    write_json(output/'runs.json',[dict(r,status='completed') for r in master['runs']])
    write_json(output/'worker_devices.json',{'devices':devices,'grouping':'condition/noise/method; all CNN checkpoints share one generated cache',
                                           'cache_policy':'release after every complete group'})
    audited_runs(output)
    return output


def ablation(root, config, selection, checkpoints, device, output):
    from src.common import digest, read_csv, write_csv, write_json, config_id
    from src.data import subset_rows
    from src.experiment import cache_images, predict, load_cnn, release_image_cache
    from src.enhancement import METHODS
    from src.metrics import classification
    root,output=Path(root),Path(output)
    manifest=root/'data/manifests/gtsrb_samples.csv'
    rows=subset_rows(read_csv(manifest),'test',10,42)
    models=[load_cnn(p,digest(manifest),device,config['cnn']) for p in checkpoints]
    base=config['conditions']['medium']
    variants={'gamma_only':{'gamma_dark':base['gamma_dark']},
              'gamma_gaussian':{'gamma_dark':base['gamma_dark'],'gaussian_sigma':base['gaussian_sigma']},
              'gamma_poisson':{'gamma_dark':base['gamma_dark'],'poisson_peak':base['poisson_peak']},
              'gamma_both':base}
    write_json(output/'protocol.json',{'role':'supplemental ablation; no training/tuning','variants':variants,
               'sample_ids':[r['sample_id'] for r in rows],'checkpoint_hashes':{p.name:digest(p) for p in checkpoints}})
    results=[]
    for variant,parameters in variants.items():
        cfg=deepcopy(config);cfg['conditions']={'medium':parameters};cfg['enhancers'].update(selection['enhancers'])
        for noise in cfg['noise_seeds']:
            for method in METHODS:
                cached=cache_images(root,rows,digest(manifest),cfg,'medium',noise,method,cfg['enhancers'][method],device)
                for model,saved in models:
                    predictions=predict(root,rows,cached,model,cfg,device)
                    score=classification([r['class_id'] for r in predictions],[r['prediction'] for r in predictions])
                    run={'variant':variant,'method':method,'noise_seed':noise,'train_seed':saved['seed']}
                    rid=config_id(run)
                    write_csv(output/'predictions'/f'{rid}.csv',predictions)
                    results.append({**run,'run_id':rid,'n_images':len(rows),**{k:score[k] for k in ['accuracy','macro_f1','weighted_f1']}})
                write_csv(output/'summary.csv',results)
                release_image_cache(root,cached)
                print(f'Ablation {len(results)}/288: {variant} {method}',flush=True)
    if len(results)!=288:raise ValueError('Incomplete ablation')
    return output


def train_fresh(root, config, output, devices):
    from concurrent.futures import ThreadPoolExecutor
    from src.training import train
    # Independent seeds can train simultaneously; no DDP, no change to each seed's batch/optimizer.
    def one(index,seed):
        command=[sys.executable,'-m','src.tools.fresh_run','train-worker','--root',str(root),
                 '--output',str(output),'--device',devices[index%len(devices)],'--seed',str(seed)]
        subprocess.run(command,cwd=root,check=True)
    # Assign each device its sequential seed queue to prevent two jobs sharing one GPU.
    def queue(rank):
        for index,seed in enumerate(config['train_seeds']):
            if index%len(devices)==rank:one(index,seed)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(queue,range(len(devices))))
    return [Path(output)/f'cnn_{s}.pt' for s in config['train_seeds']]


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=['train-worker','evaluate-worker'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--seed',type=int)
    p.add_argument('--protocol',type=Path)
    a=p.parse_args()
    from src.common import device_for
    import torch
    torch.set_num_threads(2)
    if a.task=='train-worker':
        from src.training import train
        cfg=json.loads((a.root/'configs/resnet18.json').read_text())
        train(a.root,cfg,a.seed,device_for(a.device),a.output)
    else:
        from src.experiment import evaluate
        evaluate(a.root,a.protocol,a.output,device_for(a.device),release_cache=True)
