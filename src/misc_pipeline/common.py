"""Paths and small helpers shared by every stage."""
import csv, hashlib, json, os, re, uuid
from pathlib import Path


def _find_root():
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / 'configs').is_dir() and (p / 'src' / 'misc_pipeline').is_dir():
            return p
    raise FileNotFoundError('repository root not found')


ROOT = _find_root()
DATA = Path(os.environ.get('MISC_DATA', ROOT / 'data' / 'competition_data'))   # folder with train.csv, test.csv, images/
WORK = Path(os.environ.get('MISC_WORK', ROOT / 'work'))                          # every stage writes here
RES = ROOT / 'resources'
PRIVATE = ROOT / 'private_artifacts'
STAGE_DIRS = {k: WORK / k for k in ('pass1', 'pass2', 'pass3', 'stage1', 'stage2')}


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def read_text(path):
    return Path(path).read_text(encoding='utf-8')


def prompt(name):
    """A prompt resource, byte-exact."""
    return (RES / 'prompts' / name).read_text(encoding='utf-8')


def expected_hashes():
    return json.loads((RES / 'expected_hashes.json').read_text(encoding='utf-8'))


def check_hash(label, actual, expected):
    if actual != expected:
        raise ValueError('%s changed: expected sha256 %s, got %s' % (label, expected[:16], actual[:16]))


def test_rows():
    rows = read_csv(DATA / 'test.csv')
    if len(rows) != 3498 or len({r['trace_id'] for r in rows}) != 3498:
        raise ValueError('unexpected test.csv: %d rows' % len(rows))
    return rows


def taxonomy():
    """Taxonomy records exactly as they were serialised into the stage-1 prompt (keys sorted)."""
    return [{k: r[k] for k in sorted(r)} for r in read_csv(DATA / 'taxonomy_v1.csv')]


def student_image(row):
    return DATA / 'images' / 'test' / row['image_file']


def question_image(row):
    return DATA / 'images' / 'questions' / row['question_image_file'] if row.get('question_image_file') else None
