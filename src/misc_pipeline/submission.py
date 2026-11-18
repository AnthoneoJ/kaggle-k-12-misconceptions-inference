"""Writes the Kaggle submission (trace_id,root_code; up to 3 space-separated codes per row) from the saved stage-1 and stage-2 records.
Fallback chain per row: stage-2 ranked_codes (padded with stage-1 candidates) -> stage-1 first 3 candidates -> NONE.
Usage:  python -m misc_pipeline.submission [--stage1-only] [--stage1 DIR] [--stage2 DIR] [--out FILE]   (run from src/ or with PYTHONPATH=src)"""
import argparse, csv, json
from pathlib import Path

from .common import DATA, ROOT, STAGE_DIRS


def rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load(folder):
    out = {}
    if (folder / 'records').is_dir():
        for p in (folder / 'records').glob('*.json'):
            r = json.loads(p.read_text(encoding='utf-8'))
            if r.get('status') == 'ok':
                out[r['trace_id']] = r
    return out


def make(stage1=None, stage2=None, out=None, stage1_only=False):
    S1 = Path(stage1) if stage1 else STAGE_DIRS['stage1']
    S2 = Path(stage2) if stage2 else STAGE_DIRS['stage2']
    OUT = Path(out) if out else ROOT / 'submission' / ('submission_stage1_only.csv' if stage1_only else 'submission.csv')
    sample = [r['trace_id'] for r in rows(DATA / 'sample_submission.csv')]
    test_ids = [r['trace_id'] for r in rows(DATA / 'test.csv')]
    assert len(sample) == 3498 and set(sample) == set(test_ids) and len(set(sample)) == 3498
    VALID = {r['id'] for r in rows(DATA / 'taxonomy_v1.csv')} | {'NONE'}
    R1 = load(S1); R2 = {} if stage1_only else load(S2)
    src = {'stage2': 0, 'stage2_padded': 0, 'stage1': 0, 'none': 0}; lines = []
    for tid in sample:
        c1 = [c['code'] for c in (R1.get(tid, {}).get('parsed') or {}).get('candidates', []) if c.get('code') in VALID]
        codes, kind = [], 'none'
        if tid in R2:
            codes = [c for c in R2[tid]['parsed']['ranked_codes'] if c in VALID]; kind = 'stage2'
            if len(codes) < 3:
                codes += [c for c in c1 if c not in codes]; kind = 'stage2_padded' if len(set(codes)) > len(R2[tid]['parsed']['ranked_codes']) else 'stage2'
        elif c1:
            codes, kind = c1, 'stage1'
        seen = []
        for c in codes:
            if c not in seen:
                seen.append(c)
        seen = seen[:3] or ['NONE']
        src[kind] += 1; lines.append((tid, ' '.join(seen)))
    assert [t for t, _ in lines] == sample and all(1 <= len(c.split()) <= 3 and all(x in VALID for x in c.split()) for _, c in lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['trace_id', 'root_code']); w.writerows(lines)
    print('wrote', OUT, len(lines), 'rows | sources', src)
    return OUT, src


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-only', action='store_true'); ap.add_argument('--stage1'); ap.add_argument('--stage2'); ap.add_argument('--out')
    a = ap.parse_args()
    make(a.stage1, a.stage2, a.out, a.stage1_only)
