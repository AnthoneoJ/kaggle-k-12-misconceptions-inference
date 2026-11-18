"""Download the competition data into data/competition_data (Hugging Face dataset or Kaggle), then arrange the folder layout.

Expected layout afterwards:
    data/competition_data/{train.csv,test.csv,unlabelled.csv,sample_submission.csv,taxonomy_v1.csv}
    data/competition_data/images/{train,test,unlabelled,questions}/...
The data is licensed for the competition only: never commit it or share it publicly.
"""
import os, shutil, subprocess, zipfile
from pathlib import Path

from .common import DATA, ROOT

HF_DATASET = 'colearnlab/misconception-detection'
KAGGLE_COMPETITION = 'misconception-detection-in-k-12-student-math-workings'
CSV_NAMES = ['train.csv', 'test.csv', 'unlabelled.csv', 'sample_submission.csv', 'taxonomy_v1.csv']


def _extract_zips(folder):
    for z in sorted(Path(folder).rglob('*.zip')):
        print('extracting', z.name, '...', flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent)


def organise(source, dest=None):
    """Find the folder under `source` that holds test.csv and images/, and move its content to `dest`."""
    dest = Path(dest or DATA); source = Path(source)
    csv_roots = [p.parent for p in source.rglob('test.csv')]
    img_roots = [p for p in source.rglob('images') if (p / 'test').is_dir()]
    if not csv_roots or not img_roots:
        raise FileNotFoundError('could not find test.csv and images/test under %s; check the download' % source)
    csv_dir, img_dir = csv_roots[0], img_roots[0]
    dest.mkdir(parents=True, exist_ok=True)
    for n in CSV_NAMES:
        if (csv_dir / n).is_file() and not (dest / n).is_file():
            shutil.move(str(csv_dir / n), str(dest / n))
    if not (dest / 'images').exists():
        shutil.move(str(img_dir), str(dest / 'images'))
    return dest


def from_huggingface(token=None):
    """Gated dataset: accept the conditions on the dataset page first, then provide an access token (HF_TOKEN or login prompt)."""
    from huggingface_hub import login, snapshot_download
    token = token or os.environ.get('HF_TOKEN')
    if token:
        login(token=token)
    else:
        login()   # interactive prompt
    tmp = ROOT / 'data' / '_download_hf'
    snapshot_download(repo_id=HF_DATASET, repo_type='dataset', local_dir=str(tmp))
    _extract_zips(tmp)
    out = organise(tmp)
    return out


def from_kaggle():
    """Needs the Kaggle command line tool with credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY) and accepted competition rules."""
    tmp = ROOT / 'data' / '_download_kaggle'
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(['kaggle', 'competitions', 'download', '-c', KAGGLE_COMPETITION, '-p', str(tmp)], check=True)
    _extract_zips(tmp)
    return organise(tmp)


def cleanup_downloads():
    for d in ('_download_hf', '_download_kaggle'):
        p = ROOT / 'data' / d
        if p.exists():
            shutil.rmtree(p)
