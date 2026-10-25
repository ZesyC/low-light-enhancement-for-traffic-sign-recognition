"""Restore numeric artifacts from a results ZIP, never execute bundled source."""
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import torch

from src.common import config_id,digest,write_json
from src.experiment import manifest_rows,source_code_hash
from src.model import cnn_config
from .research_evidence import audited_runs,read_csv
from src.metrics import classification


def _split_predictions(z,name,target,allowed,expected_sha=None):
    """Verify the merged CSV while splitting it, with one output file open at a time."""
    h=hashlib.sha256()
    class Reader(io.RawIOBase):
        def readable(self):return True
        def readinto(self,buffer):
            data=binary.read(len(buffer));h.update(data);buffer[:len(data)]=data;return len(data)
    current=None;stream=None;written=set()
    try:
        with z.open(name) as binary,io.TextIOWrapper(io.BufferedReader(Reader()),encoding='utf-8-sig',newline='') as text:
            for row in csv.DictReader(text):
                rid=row['run_id']
                if rid not in allowed:raise ValueError('Unexpected prediction run')
                if rid!=current:
                    if rid in written:raise ValueError('Noncontiguous/duplicate prediction run')
                    if stream is not None:stream.close()
                    current=rid;written.add(rid)
                    path=Path(target)/'predictions'/f'{rid}.csv';path.parent.mkdir(parents=True,exist_ok=True)
                    stream=path.open('w',encoding='utf-8',newline='')
                    fields=[k for k in row if k not in {'run_id','variant','condition','method','train_seed'}]
                    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
                writer.writerow(row)
    finally:
        if stream is not None:stream.close()
    if written!=allowed:raise ValueError('Missing prediction runs')
    if expected_sha and h.hexdigest()!=expected_sha:raise ValueError('Prediction checksum mismatch')


def restore(root,archive,output,config):
    root,archive,output=Path(root),Path(archive),Path(output)
    _,manifest_hash=manifest_rows(root)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive paths')
        hashes=json.loads(z.read('SHA256.json')) if 'SHA256.json' in names else {}
        def read(name):
            data=z.read(name)
            if hashes and hashlib.sha256(data).hexdigest()!=hashes.get(name):
                raise ValueError('Archive checksum mismatch: '+name)
            return data
        def save(name,path):
            data=read(name);path=Path(path)
            if path.exists() and path.read_bytes()!=data:raise ValueError('Refusing to overwrite different artifact: '+str(path))
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        checkpoints=[]
        for seed in [11,22,33]:
            candidates=[n for n in [f'05_training/cnn_{seed}.pt',f'notebook_A/train/cnn_{seed}.pt'] if n in names]
            if len(candidates)!=1:raise ValueError(f'Need exactly one checkpoint for seed {seed}')
            data=read(candidates[0])
            saved=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
            if (saved['seed']!=seed or saved['manifest_sha256']!=manifest_hash
                or cnn_config(saved.get('cnn_config'))!=cnn_config(config['cnn']) or saved.get('pilot',False)):
                raise ValueError('Checkpoint seed/data/config incompatible')
            path=output/'train'/f'cnn_{seed}.pt';save(candidates[0],path);checkpoints.append(path)
        selection_names=[n for n in ['04_protocol/validation/selection.json','notebook_A/validation/selection.json'] if n in names]
        selection_ready=False
        if len(selection_names)==1:
            selection=json.loads(read(selection_names[0]))
            if set(selection['checkpoints'].values())!={digest(p) for p in checkpoints}:
                raise ValueError('Validation/checkpoint hashes differ')
            if (selection['manifest_sha256']==manifest_hash and selection['code_hash']==source_code_hash()
                and selection.get('status')=='completed' and not selection.get('pilot',True)):
                selection['checkpoints']={str(p.resolve()):digest(p) for p in checkpoints}
                target=output/'validation/selection.json'
                if target.exists() and json.loads(target.read_text())!=selection:raise ValueError('Existing selection differs')
                write_json(target,selection);selection_ready=True
                for name,destination in [
                    ('07_supplemental/validation/validation_runs.csv','validation/validation_runs.csv'),
                    ('07_supplemental/validation/predictions.csv','validation/restored_predictions.csv')]:
                    if name in names:save(name,output/destination)
        for seed in [11,22,33]:
            for filename in [f'train_{seed}.csv',f'learning_{seed}.csv',f'train_{seed}.json',f'validation_{seed}.json']:
                candidates=[n for n in ['05_training/'+filename,'notebook_A/train/'+filename] if n in names]
                if len(candidates)==1:save(candidates[0],output/'train'/filename)
        recovered=[]
        # Readable archive: restore merged predictions to the ordinary audited run layout.
        for scope,prefix,summary,predictions in [
            ('fulltest','04_protocol/','01_summary/runs.csv','03_predictions/test.csv'),
            ('subset','07_supplemental/subset/','07_supplemental/subset/summary.csv','07_supplemental/subset/predictions.csv')]:
            if prefix+'protocol.json' not in names or summary not in names or predictions not in names:continue
            protocol=json.loads(read(prefix+'protocol.json'))
            if (protocol['manifest_sha256']!=manifest_hash or protocol['code_hash']!=source_code_hash()
                or {c['sha256'] for c in protocol['checkpoints']}!={digest(p) for p in checkpoints}):continue
            actual='fulltest' if protocol['subset_id']=='official_test' else 'subset'
            target=output/actual
            save(prefix+'protocol.json',target/'protocol.json');save(prefix+'runs.json',target/'runs.json')
            save(summary,target/'summary.csv')
            allowed={r['run_id'] for r in csv.DictReader(io.StringIO(read(summary).decode('utf-8-sig')))}
            if allowed!={config_id(r) for r in protocol['runs']}:
                raise ValueError('Prediction run IDs differ from frozen protocol')
            _split_predictions(z,predictions,target,allowed,hashes.get(predictions))
            audited_runs(target);recovered.append(actual)
        ablation_ready=False
        prefix='07_supplemental/ablation/'
        if all(prefix+n in names for n in ['protocol.json','summary.csv','predictions.csv']):
            protocol=json.loads(read(prefix+'protocol.json'))
            if set(protocol['checkpoint_hashes'].values())!={digest(p) for p in checkpoints}:
                raise ValueError('Ablation checkpoint mismatch')
            target=output/'ablation';save(prefix+'protocol.json',target/'protocol.json');save(prefix+'summary.csv',target/'summary.csv')
            summaries=read_csv(target/'summary.csv');allowed={r['run_id'] for r in summaries}
            for r in summaries:
                run={k:r[k] for k in ['variant','method','noise_seed','train_seed']}
                run['noise_seed']=int(run['noise_seed']);run['train_seed']=int(run['train_seed'])
                if config_id(run)!=r['run_id']:raise ValueError('Invalid ablation run ID')
            if len(allowed)!=288:raise ValueError('Incomplete ablation matrix')
            _split_predictions(z,prefix+'predictions.csv',target,allowed,hashes.get(prefix+'predictions.csv'))
            for r in summaries:
                rows=read_csv(target/'predictions'/f"{r['run_id']}.csv")
                if len(rows)!=430 or {p['sample_id'] for p in rows}!=set(protocol['sample_ids']):raise ValueError('Ablation coverage differs')
                if any(p['noise_seed']!=r['noise_seed'] for p in rows):raise ValueError('Ablation noise seed differs')
                score=classification([int(p['class_id']) for p in rows],[int(p['prediction']) for p in rows])
                if any(abs(score[k]-float(r[k]))>1e-8 for k in ['accuracy','macro_f1','weighted_f1']):raise ValueError('Ablation metric differs')
            ablation_ready=True
    record={'archive_sha256':digest(archive),'new_training':False,'selection_reusable':selection_ready,
        'baseline_scopes':sorted(set(recovered)),'ablation_reusable':ablation_ready,'checkpoints':{str(p):digest(p) for p in checkpoints},
        'bundled_source_executed':False,'note':'Changed baseline implementation requires validation again, never automatic retraining.'}
    write_json(output/'restore.json',record)
    return checkpoints,record
