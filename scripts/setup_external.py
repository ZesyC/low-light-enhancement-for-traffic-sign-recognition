"""Fetch pinned author code and only the three checkpoints used by this project."""
from pathlib import Path
import subprocess

import gdown

from src.common import digest, git_commit, write_json

REPOS = [
    ('Zero-DCE', 'https://github.com/Li-Chongyi/Zero-DCE.git', 'e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd'),
    ('Retinexformer', 'https://github.com/caiyuanhao1998/Retinexformer.git', '1e9a0efce4b306b6701b824768370ff26066c32a'),
    ('Diffusion-Low-Light', 'https://github.com/JianghaiSCU/Diffusion-Low-Light.git', 'e3223354f47874d3e80d969d3ab2a516874785e8'),
]


def main():
    records = []
    for name, url, commit in REPOS:
        folder = Path('external') / name
        if not folder.exists():
            folder.mkdir(parents=True)
            subprocess.run(['git', 'init', str(folder)], check=True)
            subprocess.run(['git', '-C', str(folder), 'remote', 'add', 'origin', url], check=True)
            subprocess.run(['git', '-C', str(folder), 'fetch', '--depth', '1', 'origin', commit], check=True)
            subprocess.run(['git', '-C', str(folder), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        if git_commit(folder) != commit:
            raise ValueError(f'{folder}: existing checkout differs; use a separate checkout, do not reset local changes')
        records.append({'repo': url, 'commit': commit})
    retinex_path = Path('external/Retinexformer/pretrained_weights/LOL_v1.pth')
    if not retinex_path.exists():
        files = gdown.download_folder(id='1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV', skip_download=True, quiet=True)
        target = next(f for f in files if f.path == 'LOL_v1.pth')
        retinex_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = retinex_path.with_suffix('.part')
        gdown.download(id=target.id, output=str(temporary))
        temporary.replace(retinex_path)
    diffusion_path = Path('external/Diffusion-Low-Light/ckpt/model.pth.tar')
    if not diffusion_path.exists():
        diffusion_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = diffusion_path.with_suffix('.part')
        gdown.download(id='1f4zDvPsWKrID33OJdeHwc5VOBILkm0KW', output=str(temporary))
        temporary.replace(diffusion_path)
    paths = [Path('external/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth'), retinex_path, diffusion_path]
    write_json('external/sources.json', {'repositories': records,
               'checkpoints': [{'path': str(p), 'sha256': digest(p)} for p in paths],
               'weight_sources': ['Zero-DCE author git snapshot',
                                  'https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV',
                                  'https://drive.google.com/file/d/1f4zDvPsWKrID33OJdeHwc5VOBILkm0KW/view'],
               'license_notice': 'External code/weights retain author terms; see each repository. Do not redistribute GTSRB based on library licenses.'})
    print('Author code and checkpoints ready; run python -m scripts.run smoke')


if __name__ == '__main__':
    main()
