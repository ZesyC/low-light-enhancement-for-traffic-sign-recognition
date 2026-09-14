"""Create visual audit sheets for dHash candidates; never changes the dataset."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.common import read_csv, read_image, write_json


def main():
    rows={r['sample_id']:r for r in read_csv('data/manifests/gtsrb_samples.csv')}
    candidates=json.loads(Path('data/manifests/near_duplicate_candidates.json').read_text())
    pairs=[]
    for group in candidates:
        active=[r for r in group if r['split']!='quarantine']
        if len({r['split'] for r in active})<2:
            continue
        first=active[0]
        second=next(r for r in active if r['split']!=first['split'])
        pairs.append([first['sample_id'],second['sample_id']])
    folder=Path('outputs/audit_review')
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'near_pairs.json',pairs)
    for start in range(0,len(pairs),24):
        page=Image.new('RGB',(1280,960),'white')
        draw=ImageDraw.Draw(page)
        for offset,pair in enumerate(pairs[start:start+24]):
            x,y=(offset%4)*320,(offset//4)*160
            for j,sample in enumerate(pair):
                row=rows[sample]
                crop=read_image(row['crop_path'])
                # Display gamma only, to inspect dark sources. Canonical pixels are untouched.
                display=np.rint(np.sqrt(crop)*255).astype(np.uint8)
                page.paste(Image.fromarray(display).resize((120,120)),(x+j*150,y))
                draw.text((x+j*150,y+120),f"{start+offset}: {row['split']} C{row['class_id']}",fill='black')
                draw.text((x+j*150,y+136),sample.split('/')[-1],fill='black')
        page.save(folder/f'near_pairs_{start//24+1}.png')
    assert len(pairs)<=len(candidates) and all(rows[a]['split']!=rows[b]['split'] for a,b in pairs)
    print('Active cross-split candidate pairs:',len(pairs))


if __name__=='__main__':
    main()
