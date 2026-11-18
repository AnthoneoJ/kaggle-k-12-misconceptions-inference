"""Per-row inputs: test.csv + question reference workings + earlier-stage outputs."""
import json, re

from .common import PRIVATE, RES, STAGE_DIRS, check_hash, expected_hashes, question_image, read_csv, sha256_file, student_image, test_rows

REF_FIELDS = ['question_id', 'question_text', 'schema_working', 'assumptions_or_issues', 'answer_key_consistency',
              'computed_answer', 'correct_answer', 'diagram_observations', 'source_variants', 'reference_usage_note']


def reference_cache():
    path = RES / 'reference' / 'question_solution_cache.json'
    if not path.is_file():
        path = PRIVATE / 'reference' / 'question_solution_cache.json'
    if not path.is_file():
        raise FileNotFoundError('question_solution_cache.json not found in resources/reference or private_artifacts/reference (see README, "Private artifacts")')
    check_hash('reference cache', sha256_file(path), expected_hashes()['reference_cache_sha256'])
    return json.loads(path.read_text(encoding='utf-8'))


def reference_entry(ref):
    selected = {k: ref.get(k) for k in REF_FIELDS}
    assert selected['schema_working'].strip()
    caveats = {k: selected[k] for k in ['assumptions_or_issues', 'answer_key_consistency', 'computed_answer', 'correct_answer']}
    caveats['diagram_observations'] = selected['diagram_observations'] or ''
    if selected['source_variants']:
        caveats['source_variants'] = selected['source_variants']
    if selected['reference_usage_note']:
        caveats['reference_usage_note'] = selected['reference_usage_note']
    return {'reference_working': selected['schema_working'], 'reference_caveats': json.dumps(caveats, ensure_ascii=False, indent=2)}


def _same(a, b):
    norm = lambda t: ' '.join(t.split()); unesc = lambda t: t.replace('\\n', '\n')
    return norm(a) == norm(b) or norm(unesc(a)) == norm(unesc(b))


def fixed_question(row, ref):
    """test.csv question text; the only fix is a stray trailing backtick (one row), anything else that differs raises."""
    text = row['question_text']
    if _same(text, ref['question_text']):
        return text
    stripped = re.sub(r'`+\s*$', '', text)
    if stripped != text and _same(stripped, ref['question_text']):
        return stripped
    raise ValueError('question text differs from the reference cache: ' + row['trace_id'])


def _by_id(path, col):
    return {r['trace_id']: r[col] for r in read_csv(path)} if path.is_file() else {}


def load_cases(with_pass2=False):
    cache = reference_cache()
    p1 = _by_id(STAGE_DIRS['pass1'] / 'transcriptions.csv', 'ocr')
    p2 = _by_id(STAGE_DIRS['pass2'] / 'transcriptions.csv', 'ocr') if with_pass2 else {}
    notes = _by_id(STAGE_DIRS['pass2'] / 'review_notes.csv', 'review_notes') if with_pass2 else {}
    cases = []
    for row in test_rows():
        tid, qid = row['trace_id'], row['question_id']
        ref = cache[qid]
        entry = reference_entry(ref)
        qimg = question_image(row)
        cases.append({'trace_id': tid, 'question_id': qid, 'question': fixed_question(row, ref), 'answer': row['correct_answer'],
                      'reference_working': entry['reference_working'], 'reference_caveats': entry['reference_caveats'],
                      'student_image': student_image(row), 'question_image': qimg,
                      'question_asset': (tid + '_question') if qimg else '', 'student_asset': tid + '_student',
                      'pass1': p1.get(tid), 'pass2': p2.get(tid), 'review_notes': notes.get(tid)})
    if with_pass2:
        missing = [c['trace_id'] for c in cases if c['pass1'] is None or c['pass2'] is None or c['review_notes'] is None]
        if missing:
            raise RuntimeError('pass 1/2 outputs are incomplete (%d rows, first %s); finish notebooks 01 and 02 first' % (len(missing), missing[:3]))
    return cases
