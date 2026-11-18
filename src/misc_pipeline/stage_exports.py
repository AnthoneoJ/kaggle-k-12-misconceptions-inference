"""Exports for stages 1 and 2 (results.csv and a small summary)."""
from .common import STAGE_DIRS, atomic, write_csv
from .runner import load_records


def export_stage1(jobs, metas):
    out = STAGE_DIRS['stage1']; recs = load_records(out); rows = []
    for j in jobs:
        r = recs.get(j['trace_id'])
        if not r or r.get('status') != 'ok':
            continue
        listed = [c['code'] for c in r['parsed']['candidates']]
        rows.append({'trace_id': j['trace_id'], 'n_candidates': len(listed), 'candidates': ' '.join(listed), 'used_pass3': metas[j['trace_id']]['used_pass3'],
                     'completion_tokens': (r.get('usage') or {}).get('completion_tokens', ''), 'seconds': round(r.get('seconds', 0), 1)})
    if rows:
        write_csv(out / 'results.csv', rows)
    s = {'expected': len(jobs), 'completed': len(rows), 'complete': len(rows) == len(jobs)}
    atomic(out / 'export_summary.json', s)
    return s


def export_stage2(jobs, metas, n_rows_total):
    out = STAGE_DIRS['stage2']; recs = load_records(out); rows = []
    for j in jobs:
        tid = j['trace_id']; r = recs.get(tid)
        if not r or r.get('status') != 'ok':
            continue
        p = r['parsed']; listed = metas[tid]['listed']
        rows.append({'trace_id': tid, 'ranked_codes': ' '.join(p['ranked_codes']), 'short': p['short'], 'extra_codes': ' '.join(p['extra_codes']),
                     'stage1_listed': ' '.join(listed), 'stage1_top3': ' '.join(listed[:3]), 'agrees_top1': p['ranked_codes'][0] == listed[0],
                     'used_pass3': metas[tid]['used_pass3'], 'pass3_changed': metas[tid]['changed'],
                     'completion_tokens': (r.get('usage') or {}).get('completion_tokens', ''), 'seconds': round(r.get('seconds', 0), 1)})
    if rows:
        write_csv(out / 'results.csv', rows)
    s = {'expected': n_rows_total, 'jobs_built': len(jobs), 'completed': len(rows), 'complete': len(rows) == n_rows_total,
         'short_answers': sum(bool(r['short']) for r in rows)}
    atomic(out / 'export_summary.json', s)
    return s
