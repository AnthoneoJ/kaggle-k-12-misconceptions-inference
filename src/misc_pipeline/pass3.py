"""Pass 3: text-only coherency check of the pass-2 transcription (question diagram attached, no student photo)."""
import hashlib, json, re

from . import inputs, pass2
from .common import STAGE_DIRS, atomic, check_hash, expected_hashes, prompt, write_csv
from .marks import collapse, new_marks, norm, split_sections
from .runner import load_records

NAME = 'pass3'
BLOCKS = [('question', 'QUESTION'), ('reference_working', 'REFERENCE_WORKING'), ('reference_caveats', 'REFERENCE_CAVEATS'),
          ('pass1_transcription', 'PASS-1_TRANSCRIPTION'), ('pass2_transcription', 'PASS-2_TRANSCRIPTION')]


def _tagged(text, tag):
    m = re.search(r'<%s>\n?(.*?)\n?</%s>' % (tag, tag), text, re.S)
    if not m:
        raise ValueError('missing <%s> in the pass-2 prompt' % tag)
    return m.group(1).strip('\n')


def load_pass2_texts(rows):
    out = {}
    for tid in rows:
        p = STAGE_DIRS['pass2'] / 'records' / (tid + '.json')
        if p.exists():
            r = json.loads(p.read_text(encoding='utf-8'))
            text = ((r.get('parsed') or {}).get('ocr') or '').strip('\n')
            if r.get('status') == 'ok' and text.strip():
                out[tid] = text
    return out


def build_jobs(limit=None):
    base = prompt('pass3_coherency_v4.txt').strip('\n')
    check_hash('pass-3 prompt', hashlib.sha256(base.encode()).hexdigest(), expected_hashes()['pass3_prompt_sha256'])
    cases = inputs.load_cases()
    if limit:
        cases = cases[:int(limit)]
    p2_jobs = {j['trace_id']: j for j in pass2.build_jobs(limit)}
    p2 = load_pass2_texts([c['trace_id'] for c in cases])
    missing = [c['trace_id'] for c in cases if c['trace_id'] not in p2]
    if missing:
        raise RuntimeError('pass 2 is incomplete (%d rows without an ok record); finish notebook 02 first' % len(missing))
    jobs = []
    for c in cases:
        tid = c['trace_id']
        pr = p2_jobs[tid]['prompt']
        inp = {'question': _tagged(pr, 'question'), 'reference_working': _tagged(pr, 'reference_working'),
               'reference_caveats': _tagged(pr, 'reference_caveats'), 'pass1_transcription': _tagged(pr, 'initial_transcription'),
               'pass2_transcription': p2[tid]}
        assert inp['pass1_transcription'] == c['pass1'].strip('\n'), 'pass-1 text differs from the pass-1 CSV: ' + tid
        text = base + '\n\n' + '\n\n'.join('<%s>\n%s\n</%s>' % (tag, inp[k], tag) for k, tag in BLOCKS)
        assert 'PASS-2_REVIEW_NOTES' not in text
        images = [('question_diagram', c['question_image'])] if c['question_image'] else []
        jobs.append({'trace_id': tid, 'prompt': text, 'images': images})
    return jobs


def parse(rec):
    return split_sections(rec['content'])


def export(jobs):
    out = STAGE_DIRS[NAME]; recs = load_records(out); p2 = load_pass2_texts([j['trace_id'] for j in jobs])
    hand, rows = {}, []
    for j in jobs:
        tid = j['trace_id']; r = recs.get(tid)
        if not r or r.get('status') != 'ok' or tid not in p2:
            continue
        p3 = r['parsed']['transcription']; nm = new_marks(p3, p2[tid])
        silent = norm(collapse(p3, 1)) != norm(collapse(p2[tid], 1))
        hand[tid] = {'transcription': p3, 'transcription_resolved': collapse(p3, 0), 'pass2_transcription': p2[tid], 'new_marks': nm,
                     'silent_edit_outside_marks': silent, 'review_notes': r['parsed']['review_notes']}
        rows.append({'trace_id': tid, 'n_new_marks': len(nm), 'new_marks': ' | '.join(nm), 'silent_edit_outside_marks': silent,
                     'completion_tokens': (r.get('usage') or {}).get('completion_tokens', ''), 'seconds': round(r.get('seconds', 0), 1)})
    atomic(out / 'pass3_transcriptions.json', hand)
    if rows:
        write_csv(out / 'results.csv', rows)
    summary = {'expected': len(jobs), 'completed': len(hand), 'complete': len(hand) == len(jobs),
               'rows_with_new_mark': sum(r['n_new_marks'] > 0 for r in rows), 'silent_edit_rows': sum(r['silent_edit_outside_marks'] for r in rows)}
    atomic(out / 'export_summary.json', summary)
    return summary
