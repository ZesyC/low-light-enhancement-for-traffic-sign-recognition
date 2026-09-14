"""Summarize already-cached validation inputs without accessing test predictions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.common import read_csv, write_csv
from src.experiment import manifest_rows, source_code_hash
from src.data import subset_rows


def main():
    root=Path('.')
    config=json.loads(Path('configs/default.json').read_text())
    samples, manifest_hash=manifest_rows(root)
    ids=[r['sample_id'] for r in subset_rows(samples,'validation',30)]
    records=[]
    for path in sorted((root/'data/cache').glob('*/metadata.json')):
        meta=json.loads(path.read_text())
        if (meta['status']=='completed' and meta['enhancer']['method']=='identity'
            and meta['manifest_sha256']==manifest_hash and meta['code_hash']==source_code_hash()
            and meta['sample_ids']==ids):
            rows=read_csv(path.with_name('images.csv'))
            assert len(rows)==len(ids) and [r['sample_id'] for r in rows]==ids
            records.append({'condition':meta['condition'],'noise_seed':meta['noise_seed'],'n_images':len(rows),
                            **{k:float(np.mean([float(r[k]) for r in rows])) for k in
                               ['input_mean','input_clipped_fraction','psnr','ssim']},
                            'fraction_images_mean_below_0.01':sum(float(r['input_mean'])<0.01 for r in rows)/len(rows)})
    assert len(records)==len(config['conditions'])*len(config['noise_seeds'])
    write_csv('outputs/validation_degradation.csv',records)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,key,label in zip(axes,['input_mean','psnr','ssim'],['Mean RGB [0,1]','PSNR (dB)','SSIM']):
        levels=list(config['conditions'])
        means=[np.mean([r[key] for r in records if r['condition']==c]) for c in levels]
        ax.bar(levels,means,color=['#799bc3','#416d9f','#173e66'])
        ax.set(ylabel=label)
    fig.suptitle('1,290 validation crops; average over 3 noise seeds')
    fig.tight_layout()
    fig.savefig('outputs/validation_degradation.png',dpi=160)
    plt.close(fig)
    print(records)


if __name__=='__main__':
    main()
