"""Fetch upstream code/checkpoints directly; never upload or relicense those assets."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request


def provision(root, specification):
    from src.common import digest
    root = Path(root).resolve()
    spec = json.loads(Path(specification).read_text(encoding='utf-8'))
    records = []
    for item in spec['models']:
        target = root / item['directory']
        if not target.exists():
            subprocess.run(['git','clone','--quiet','--depth','1',item['repository'],str(target)],check=True)
        actual = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        if actual != item['commit']:
            subprocess.run(['git','-C',str(target),'fetch','--quiet','--depth','1','origin',item['commit']],check=True)
            subprocess.run(['git','-C',str(target),'checkout','--quiet','--detach',item['commit']],check=True)
        checkpoint = root / item['checkpoint']
        if not checkpoint.exists():
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            if item['download']['kind'] == 'gdrive':
                import gdown
                result = gdown.download(id=item['download']['id'],output=str(checkpoint),quiet=True)
                if not result:
                    raise RuntimeError(f"Author checkpoint unavailable: {item['name']}")
            else:
                with urllib.request.urlopen(item['download']['url'],timeout=120) as response, checkpoint.open('wb') as stream:
                    import shutil
                    shutil.copyfileobj(response,stream)
        sha = digest(checkpoint)
        if item.get('sha256') and sha != item['sha256']:
            raise ValueError(f"Checkpoint checksum mismatch: {item['name']}")
        records.append({'method':item['name'],'repository':item['repository'],'commit':item['commit'],
                        'checkpoint_sha256':sha,'source':item['source_page'],'license_note':item['license_note']})
        print('Verified upstream asset:',item['name'],sha,flush=True)
    return records


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--spec',type=Path,default=Path('configs/public_sources.json'))
    a=p.parse_args()
    print(json.dumps(provision(a.root,a.spec),indent=2))
