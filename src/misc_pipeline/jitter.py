"""Second submission: a small post-processing rule on top of the stage-2 ranking (notebook 07).

Rule: if NONE is in the top 2 of stage 1 (its first two candidates) AND in the top 2 of stage 2, and ANS-01 is not in the top 2 of stage 1,
then NONE becomes stage-2 place 1; the other codes keep their order. Rows without both stage records are left unchanged.

Reads submission/submission.csv (notebook 06) and the stage-1 and stage-2 records. Writes submission/submission_jitter.csv and submission/jitter_changes.csv.
Usage:  PYTHONPATH=src python -m misc_pipeline.jitter
"""
import csv

from .common import DATA, ROOT, STAGE_DIRS
from .submission import load, rows


def rule(s1, s2):
    t1, t2 = s1[:2], s2[:2]
    if 'NONE' in t1 and 'NONE' in t2 and 'ANS-01' not in t1 and s2[0] != 'NONE':
        return ['NONE'] + [c for c in s2 if c != 'NONE']
    return s2


def make(base=None, out=None):
    base = base or ROOT / 'submission' / 'submission.csv'
    out = out or ROOT / 'submission' / 'submission_jitter.csv'
    R1, R2 = load(STAGE_DIRS['stage1']), load(STAGE_DIRS['stage2'])
    sample = [r['trace_id'] for r in rows(DATA / 'sample_submission.csv')]
    valid = {r['id'] for r in rows(DATA / 'taxonomy_v1.csv')} | {'NONE'}
    sub = rows(base)
    assert [r['trace_id'] for r in sub] == sample, 'base submission does not match sample_submission.csv; run notebook 06 first'
    lines, changes = [], []
    for r in sub:
        t, s2 = r['trace_id'], r['root_code'].split()
        new = s2
        if t in R1 and t in R2:
            s1 = [c['code'] for c in (R1[t].get('parsed') or {}).get('candidates', [])]
            new = rule(s1, s2)
            if new != s2:
                changes.append((t, ' '.join(s1[:2]), ' '.join(s2), ' '.join(new[:3])))
        lines.append((t, ' '.join(new[:3])))
    assert all(1 <= len(c.split()) <= 3 and set(c.split()) <= valid for _, c in lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['trace_id', 'root_code']); w.writerows(lines)
    ch = out.with_name('jitter_changes.csv')
    with open(ch, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['trace_id', 'stage1_top2', 'stage2_before', 'stage2_after']); w.writerows(changes)
    print('wrote', out, len(lines), 'rows | changed', len(changes))
    return out, changes


if __name__ == '__main__':
    make()
