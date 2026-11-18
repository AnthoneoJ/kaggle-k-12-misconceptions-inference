"""Pass 1: image-only transcription of the student working."""
from .common import STAGE_DIRS, prompt, student_image, test_rows, write_csv
from .runner import load_records

NAME = 'pass1'


def build_jobs(limit=None):
    text = prompt('pass1_transcribe.txt')
    rows = test_rows()[:int(limit)] if limit else test_rows()
    return [{'trace_id': r['trace_id'], 'prompt': text, 'images': [('student_working', student_image(r))]} for r in rows]


def parse(rec):
    text = rec['content']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Missing/empty transcription')
    return {'ocr': text}


def export(jobs):
    out = STAGE_DIRS[NAME]; recs = load_records(out)
    qid = {r['trace_id']: r['question_id'] for r in test_rows()}
    rows = [{'trace_id': j['trace_id'], 'question_id': qid[j['trace_id']], 'ocr': recs[j['trace_id']]['parsed']['ocr']}
            for j in jobs if recs.get(j['trace_id'], {}).get('status') == 'ok']
    complete = len(rows) == len(jobs)
    write_csv(out / ('transcriptions.csv' if complete else 'transcriptions_partial.csv'), rows, ['trace_id', 'question_id', 'ocr'])
    return {'completed': len(rows), 'expected': len(jobs), 'complete': complete}
