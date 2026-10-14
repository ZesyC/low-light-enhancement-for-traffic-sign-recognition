"""Download one named output directly; avoid enumerating a notebook's image cache."""
import argparse
from pathlib import Path
from zipfile import ZipFile


def save_archive(response, target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    size = 0
    with partial.open('wb') as out:
        for chunk in response.iter_content(1024 * 1024):
            out.write(chunk)
            size += len(chunk)
    expected = response.headers.get('Content-Length')
    if expected and size != int(expected):
        raise ValueError('Incomplete Kaggle download')
    with ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError('Kaggle archive failed CRC verification')
    partial.rename(target)
    print(f'Downloaded and CRC verified: {target.name} ({size:,} bytes)', flush=True)


def main():
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel')
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    if args.archive.exists():
        with ZipFile(args.archive) as archive:
            if archive.testzip() is not None:
                raise ValueError('Existing archive failed CRC verification')
        return
    api = KaggleApi()
    api.authenticate()
    owner, slug, version = api.parse_kernel_string(args.kernel)
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug, request.kernel_slug = owner, slug
    request.file_path = args.archive.name
    if version:
        request.version_number = int(version)
    with api.build_kaggle_client() as client:
        with client.kernels.kernels_api_client.download_kernel_output(request) as response:
            response.raise_for_status()
            url = response.url
        with requests.get(url, stream=True, timeout=(20, 60)) as response:
            save_archive(response, args.archive)


if __name__ == '__main__':
    main()
