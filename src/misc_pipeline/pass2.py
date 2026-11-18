"""Pass 2: reference-examiner review of pass 1 (student photo + question diagram + reference working)."""
import re

from . import inputs
from .common import STAGE_DIRS, prompt, write_csv
from .runner import load_records

NAME = 'pass2'


def render_reference_prompt(template, values):
    # One substitution pass for four exact placeholders; never str.format().
    # Literal {{4 / 9}} markers and braces inside supplied content survive unchanged.
    required = {'question', 'reference_working', 'reference_caveats', 'transcription'}
    if set(values) != required:
        raise ValueError('Missing or extra prompt fields')
    pattern = r'(?<!\{)\{(question|reference_working|reference_caveats|transcription)\}(?!\})'
    found = re.findall(pattern, template)
    if len(found) != 4 or set(found) != required:
        raise ValueError('Expected exactly four prompt placeholders')
    return re.sub(pattern, lambda match: values[match.group(1)], template)


def build_jobs(limit=None):
    template = prompt('pass2_review_template.txt')
    cases = inputs.load_cases()
    if limit:
        cases = cases[:int(limit)]
    jobs = []
    for c in cases:
        if c['pass1'] is None:
            raise RuntimeError('pass-1 transcriptions.csv is incomplete; finish notebook 01 first')
        values = {'question': c['question'], 'reference_working': c['reference_working'],
                  'reference_caveats': c['reference_caveats'], 'transcription': c['pass1']}
        if any(not isinstance(v, str) or not v.strip() for v in values.values()):
            raise ValueError('Missing prompt content: ' + c['trace_id'])
        images = [('student_working', c['student_image'])] + ([('question_diagram', c['question_image'])] if c['question_image'] else [])
        jobs.append({'trace_id': c['trace_id'], 'question_id': c['question_id'], 'prompt': render_reference_prompt(template, values), 'images': images})
    return jobs


def parse(rec):
    text = rec['content']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty review response')
    lines = text.splitlines(keepends=True)
    positions = {'finaltranscription': [], 'reviewnotes': []}
    for index, line in enumerate(lines):
        label = re.sub(r'[\s_-]+', '', line.strip().strip('#*` :\t\r\n')).casefold()
        if label in positions:
            positions[label].append(index)
    if any(len(p) != 1 for p in positions.values()):
        raise ValueError('Need FINAL_TRANSCRIPTION and REVIEW_NOTES headings to separate student text from notes')
    first, second = positions['finaltranscription'][0], positions['reviewnotes'][0]
    if first >= second:
        raise ValueError('FINAL_TRANSCRIPTION must precede REVIEW_NOTES')
    transcription = ''.join(lines[first + 1:second])
    notes = ''.join(lines[second + 1:])
    if not transcription.strip() or not notes.strip():
        raise ValueError('Empty final transcription or review notes')
    return {'ocr': transcription, 'review_notes': notes}


def export(jobs):
    out = STAGE_DIRS[NAME]; recs = load_records(out)
    done = [j for j in jobs if recs.get(j['trace_id'], {}).get('status') == 'ok']
    complete = len(done) == len(jobs)
    sfx = '' if complete else '_partial'
    write_csv(out / ('transcriptions%s.csv' % sfx), [{'trace_id': j['trace_id'], 'question_id': j['question_id'], 'ocr': recs[j['trace_id']]['parsed']['ocr']} for j in done],
              ['trace_id', 'question_id', 'ocr'])
    write_csv(out / ('review_notes%s.csv' % sfx), [{'trace_id': j['trace_id'], 'review_notes': recs[j['trace_id']]['parsed']['review_notes']} for j in done],
              ['trace_id', 'review_notes'])
    return {'completed': len(done), 'expected': len(jobs), 'complete': complete}
