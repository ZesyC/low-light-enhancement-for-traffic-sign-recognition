"""Verify image contracts, paired selection, safe recovery and resumable inference."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

from src.common import config_id,digest,write_csv,write_json
from src.experiment import source_code_hash
from src.tools.checkpoint_restore import restore,_split_predictions
from src.tools.processing_pipeline import denoise,Pipeline,pipeline_provenance
from src.tools.processing_study import choose_fulltest_methods,evaluate,study_hash,tune,_process_group,_reference_images

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'configs/processing_extension.json').read_text())


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'tmp/tests';scratch.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch);self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def manifest(self):
        image=np.full((16,16,3),.3,np.float32)
        Image.fromarray(np.rint(image*255).astype('uint8')).save(self.root/'a.png')
        rows=[dict(sample_id='a',group_id='track-a',source_id='a',class_id=0,crop_path='a.png',split='test')]
        path=self.root/'data/manifests/gtsrb_samples.csv';write_csv(path,rows)
        write_json(path.with_suffix('.json'),{'sha256':digest(path)})
        write_json(path.parent/'audit.json',{'status':'completed'})
        return digest(path)

    def test_filters_preserve_rgb_and_reduce_flat_field_noise(self):
        clean=np.full((64,64,3),[.75,.4,.12],np.float32)
        noisy=np.clip(clean+np.random.default_rng(42).normal(0,.035,clean.shape),0,1).astype(np.float32)
        for method,configs in PLAN['denoisers'].items():
            with self.subTest(method=method):
                result=denoise(noisy,method,configs[-1])
                self.assertEqual(result.shape,clean.shape);self.assertEqual(result.dtype,np.float32)
                self.assertTrue(np.isfinite(result).all());self.assertTrue(np.all((result>=0)&(result<=1)))
                self.assertLess(np.mean((result-clean)**2),np.mean((noisy-clean)**2))
                self.assertGreater(result[...,0].mean(),result[...,2].mean()+.5)

    def test_orders_use_identical_parameters_but_different_images(self):
        image=np.random.default_rng(1).uniform(.01,.5,(32,32,3)).astype(np.float32)
        original=image.copy()
        d={'kind':'denoise','method':'gaussian','params':PLAN['denoisers']['gaussian'][1]}
        e={'kind':'enhance','method':'gamma','params':{'gamma':.5}}
        before=Pipeline({'steps':[d,e]},torch.device('cpu'))(image)
        after=Pipeline({'steps':[e,d]},torch.device('cpu'))(image)
        self.assertGreater(float(np.max(abs(before-after))),.001)
        np.testing.assert_array_equal(image,original)

    def test_invalid_filter_parameters_rejected(self):
        image=np.zeros((16,16,3),np.float32)
        for params in [{'kernel':4,'sigma':1},{'kernel':3,'sigma':float('nan')}]:
            with self.assertRaises(ValueError):denoise(image,'gaussian',params)
        with self.assertRaises(ValueError):denoise(image,'unknown',{})

    def test_validation_selects_both_orders_and_corresponding_controls(self):
        scores=[dict(method=m,category='control',macro_f1=.7,latency_ms=1) for m in ['identity','gamma','clahe']]
        scores += [dict(method='denoise__'+m,category='denoise',macro_f1=.72,latency_ms=2) for m in ['gaussian','bilateral']]
        for m,a,b in [('gaussian',.9,.5),('bilateral',.8,.79)]:
            for prefix,category,score in [('pre','denoise_then_enhance',a),('post','enhance_then_denoise',b)]:
                scores.append(dict(method=f'{prefix}__{m}__gamma',category=category,denoiser=m,macro_f1=score,latency_ms=2))
        methods=choose_fulltest_methods(scores)
        self.assertTrue({'pre__bilateral__gamma','post__bilateral__gamma','denoise__bilateral','gamma','identity'}<=set(methods))
        self.assertIn('pre__gaussian__gamma',methods)  # Global winner retained separately.

    def test_restore_rejects_missing_checkpoint_and_wrong_manifest(self):
        sha=self.manifest();config={'cnn':{}}
        for wrong in [False,True]:
            archive=self.root/f'bad-{wrong}.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for seed in ([11,22] if not wrong else [11,22,33]):
                    stream=io.BytesIO();torch.save({'seed':seed,'manifest_sha256':'wrong' if wrong else sha,'cnn_config':{},'pilot':False},stream)
                    z.writestr(f'05_training/cnn_{seed}.pt',stream.getvalue())
            with self.assertRaises(ValueError):restore(self.root,archive,self.root/f'restore-{wrong}',config)

    def test_restore_reads_weights_without_executing_bundled_source(self):
        sha=self.manifest();archive=self.root/'weights.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for seed in [11,22,33]:
                stream=io.BytesIO();torch.save({'seed':seed,'manifest_sha256':sha,'cnn_config':{},'pilot':False},stream)
                z.writestr(f'05_training/cnn_{seed}.pt',stream.getvalue())
            z.writestr('06_reproduce/evil.py',"raise RuntimeError('should never execute')")
        paths,record=restore(self.root,archive,self.root/'restored',{'cnn':{}})
        self.assertEqual(len(paths),3);self.assertFalse(record['bundled_source_executed'])
        self.assertFalse((self.root/'restored/evil.py').exists())

    def test_streamed_archive_restore_checks_checksum_and_run_ids(self):
        import csv,hashlib
        text=io.StringIO(newline='')
        fields=['run_id','method','condition','train_seed','sample_id','class_id','prediction','noise_seed']
        writer=csv.DictWriter(text,fieldnames=fields);writer.writeheader()
        for rid in ['a'*16,'b'*16]:
            writer.writerow(dict(run_id=rid,method='identity',condition='medium',train_seed=11,
                sample_id='a',class_id=0,prediction=0,noise_seed=101))
        data=text.getvalue().encode();archive=self.root/'stream.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('predictions.csv',data)
        with zipfile.ZipFile(archive) as z:
            _split_predictions(z,'predictions.csv',self.root/'split',{'a'*16,'b'*16},hashlib.sha256(data).hexdigest())
            with self.assertRaisesRegex(ValueError,'checksum'):
                _split_predictions(z,'predictions.csv',self.root/'bad-hash',{'a'*16,'b'*16},'wrong')
            with self.assertRaisesRegex(ValueError,'Unexpected'):
                _split_predictions(z,'predictions.csv',self.root/'bad-id',{'a'*16},None)
        self.assertEqual(len(list((self.root/'split/predictions').glob('*.csv'))),2)

    def test_streamed_inference_resumes_and_rejects_changed_prediction(self):
        sha=self.manifest();checkpoint=self.root/'weights.pt';checkpoint.write_bytes(b'fixture')
        config={'cnn':{'batch_size':2},'conditions':{'medium':{'gamma_dark':2}},
                'ssim':{'channel_axis':2,'data_range':1.0}}
        run=dict(condition='medium',method='identity',noise_seed=101,train_seed=11)
        spec={'steps':[]}
        protocol={'study':'denoise_order_extension','pilot':True,'sample_ids':['a'],'subset_id':'fixture',
            'manifest_sha256':sha,'code_hash':source_code_hash(),'study_hash':study_hash(),'config':config,
            'checkpoints':[dict(path=str(checkpoint),sha256=digest(checkpoint),seed=11)],
            'registry':{'identity':spec},'pipeline_provenance':{'identity':pipeline_provenance(spec)},'runs':[run]}
        path=self.root/'protocol.json';write_json(path,protocol)
        class Classifier(torch.nn.Module):
            def forward(self,images):
                result=torch.zeros((len(images),43));result[:,0]=10;return result
        out=self.root/'evaluated'
        with patch('src.tools.processing_study.load_cnn',return_value=(Classifier(),{'seed':11})) as load:
            evaluate(self.root,path,out,torch.device('cpu'));load.assert_called_once()
        with patch('src.tools.processing_study.load_cnn') as load:
            evaluate(self.root,path,out,torch.device('cpu'));load.assert_not_called()
        pred=out/'predictions'/f'{config_id(run)}.csv'
        pred.write_text(pred.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'changed'):
            evaluate(self.root,path,out,torch.device('cpu'))

    def test_validation_resume_keeps_fixed_strengths_for_both_orders(self):
        sha=self.manifest();checkpoints=[]
        for seed in [11,22,33]:
            p=self.root/f'cnn_{seed}.pt';p.write_bytes(str(seed).encode());checkpoints.append(p)
        params={'gamma':{'gamma':.5},'clahe':{'clip_limit':2,'tile_grid':[8,8]},'identity':{}}
        from src.enhancement import provenance
        baseline={'manifest_sha256':sha,'code_hash':source_code_hash(),'status':'completed','pilot':False,
            'checkpoints':{str(p):digest(p) for p in checkpoints},'sample_ids':['a'],'chosen_method':'gamma',
            'enhancers':params,'enhancer_provenance':{m:provenance(m,p) for m,p in params.items()},
            'config':{'cnn':{'batch_size':2},'conditions':{'medium':{}},'noise_seeds':[101],'enhancers':params},
            'scores':[dict(method=m,macro_f1=.1,latency_ms=1) for m in params]}
        path=self.root/'baseline.json';write_json(path,baseline)
        rows=[dict(sample_id='a',class_id=0,group_id='track-a',crop_path='a.png')]
        def infer(root,rows,cfg,condition,noise,spec,models,device,references=None):
            return {saved['seed']:[dict(sample_id='a',class_id=0,group_id='track-a',prediction=0,
                noise_seed=101,latency_ms=2,psnr=20,ssim=.5)] for _,saved in models}
        def classifier(path,*args):return None,{'seed':int(Path(path).stem.split('_')[1])}
        output=self.root/'validation'
        with patch('src.tools.processing_study.subset_rows',return_value=rows),patch('src.tools.processing_study.load_cnn',side_effect=classifier),patch('src.tools.processing_study._process_group',side_effect=infer) as process:
            selected=tune(self.root,path,checkpoints,ROOT/'configs/processing_extension.json',torch.device('cpu'),output)
            self.assertEqual(process.call_count,18)
            for m in selected['registry']:
                if m.startswith('pre__'):
                    mate='post__'+m.removeprefix('pre__')
                    self.assertEqual(selected['registry'][m]['steps'][0],selected['registry'][mate]['steps'][1])
            (output/'selection.json').unlink()
            process.reset_mock()
            resumed=tune(self.root,path,checkpoints,ROOT/'configs/processing_extension.json',torch.device('cpu'),output)
            process.assert_not_called()
            self.assertEqual(resumed['fulltest_methods'],selected['fulltest_methods'])

    def test_ram_references_preserve_predictions_and_quality_without_repeat_image_io(self):
        self.manifest()
        rows=[dict(sample_id='a',class_id=0,group_id='track-a',source_id='a',crop_path='a.png')]
        config={'cnn':{'batch_size':2},'conditions':{'medium':{'gamma_dark':2}},
                'ssim':{'channel_axis':2,'data_range':1.0}}
        class Classifier(torch.nn.Module):
            def forward(self,images):
                return images.mean((1,2,3))[:,None].repeat(1,43)+torch.arange(43)[None,:]/100
        models=[(Classifier(),{'seed':seed}) for seed in [11,22,33]]
        spec={'steps':[{'kind':'enhance','method':'gamma','params':{'gamma':.6}}]}
        normal=_process_group(self.root,rows,config,'medium',101,spec,models,torch.device('cpu'))
        references=_reference_images(self.root,rows)
        with patch('src.tools.processing_study.read_image',side_effect=AssertionError('PNG decoded again')):
            cached=_process_group(self.root,rows,config,'medium',101,spec,models,torch.device('cpu'),references)
        for seed in [11,22,33]:
            for metric in ['prediction','true_probability','psnr','ssim']:
                self.assertEqual(normal[seed][0][metric],cached[seed][0][metric])
        self.assertFalse(references['a'].flags.writeable)


if __name__=='__main__':unittest.main()
