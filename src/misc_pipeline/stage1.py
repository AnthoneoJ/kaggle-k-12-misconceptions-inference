"""Stage 1: candidate codes. Prompt assembly, parsing and job construction."""
import json, re
from pathlib import Path

from . import inputs
from .common import STAGE_DIRS, atomic, digest, prompt as load_prompt, taxonomy
from .marks import collapse, norm

INCLUDE_PASS2_NOTES = True
SILENT_EDIT_POLICY = 'fallback_pass2'      # pass-3 text is ignored for a row where pass 3 edited outside its marks
TEMPLATE = load_prompt('stage1_candidates_template.txt')


def render(template, case, taxonomy):
    return template.format(question=case['question'], answer=case['answer'],
        diagram_note='The image labelled student_working is the original student photo. ' +
            ('The separately labelled question_diagram is the supplied QUESTION diagram.' if case['question_asset'] else 'There is no separate question diagram.'),
        reference_working=case['reference_working'], reference_caveats=case['reference_caveats'],
        pass1=case['pass1'], pass2=case['pass2'], review_notes=case['review_notes'],
        taxonomy=json.dumps(taxonomy, ensure_ascii=False))


FENCE = re.compile(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n?```', re.S | re.I)


def parse(text, codes):
    """Last fenced JSON block (or a trailing bare JSON object). Invalid/duplicate codes are dropped and flagged."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty final response')
    blocks = list(FENCE.finditer(text))
    if blocks:
        raw_json, analysis = blocks[-1].group(1), text[:blocks[-1].start()]
    else:
        start = text.rfind('{"candidates"')
        if start < 0:
            start = text.find('{')
        raw_json, analysis = text[start:], text[:start]
    repaired = False
    try:
        value = json.loads(raw_json)
    except json.JSONDecodeError:
        if not raw_json.rstrip().endswith(']'):
            raise
        value = json.loads(raw_json.rstrip() + '\n}')
        repaired = True
    if not isinstance(value, dict) or not isinstance(value.get('candidates'), list):
        raise ValueError('Missing candidates list')
    kept, dropped, seen = [], [], set()
    for item in value['candidates']:
        code = item.get('code') if isinstance(item, dict) else item
        if not isinstance(code, str) or code not in codes or code in seen:
            dropped.append(code); continue
        seen.add(code)
        kept.append({'code': code, 'why': str(item.get('why', '')) if isinstance(item, dict) else '',
                     'depends_on_reading': str(item.get('depends_on_reading', '')) if isinstance(item, dict) else ''})
    if not kept:
        raise ValueError('No valid taxonomy codes')
    queries = [q for q in value.get('search_queries', []) if isinstance(q, str) and q.strip()]
    analysis = re.sub(r'^\s*#*\s*ANALYSIS\s*:?\s*', '', analysis.strip(), flags=re.I)
    result = {'candidates': kept, 'search_queries': queries, 'analysis': analysis,
              'dropped_codes': dropped, 'over_limit': len(kept) > 10}
    if repaired:
        result['json_repaired_missing_brace'] = True
    return result


_strict_parse = parse
LATEX_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')
TAIL_JSON = re.compile(r'\{\s*"candidates"')


def parse(text, codes):
    try:
        return _strict_parse(text, codes)
    except json.JSONDecodeError as exc:
        if 'escape' not in str(exc):
            raise
        result = _strict_parse(LATEX_ESCAPE.sub(r'\\\\', text), codes)
        result['json_repaired_latex_escape'] = True
        return result


def answer_text(record):
    content = record.get('content')
    if isinstance(content, str) and content.strip():
        return content, 'content'
    reasoning = record.get('reasoning') or ''
    blocks = list(FENCE.finditer(reasoning))
    if blocks and not reasoning[blocks[-1].end():].strip():
        return reasoning[blocks[-1].start():], 'reasoning_tail'
    starts = list(TAIL_JSON.finditer(reasoning))
    if starts:
        return reasoning[starts[-1].start():], 'reasoning_tail'
    return content, 'content'


def parse_record(rec):
    text, source = answer_text(rec)
    parsed = parse(text, CODES())
    if source != 'content':
        parsed['answer_source'] = source
    return parsed


def CODES():
    return [t['id'] for t in taxonomy()] + ['NONE']


def build_job(case, p3text):
    """One stage-1 job. p3text is the pass-3 transcription (or None when pass 3 gave nothing for the row)."""
    p2 = case['pass2']
    silent = p3text is not None and norm(collapse(p3text, 1)) != norm(collapse(p2, 1))
    used = p3text is not None and not (silent and SILENT_EDIT_POLICY == 'fallback_pass2')
    changed = used and p3text.strip() != p2.strip()
    case = dict(case)
    if changed:
        case['pass2'] = '\n' + p3text.strip() + '\n\n'      # the block carries the pass-3 text (unchanged rows keep the pass-2 bytes)
    text = render(TEMPLATE, case, taxonomy())
    assert 'pass_2_transcription' not in text and 'PASS-2' not in text
    images = [('student_working', case['student_image'])] + ([('question_diagram', case['question_image'])] if case['question_image'] else [])
    meta = {'pass3_present': p3text is not None, 'used_pass3': bool(used), 'changed': bool(changed), 'silent_edit': bool(silent)}
    return {'trace_id': case['trace_id'], 'question_id': case['question_id'], 'group': 'test', 'prompt': text, 'images': images}, meta


def load_pass3_texts(rows):
    """{trace_id: pass-3 transcription} for every row with an ok pass-3 record."""
    out = {}
    for tid in rows:
        p = STAGE_DIRS['pass3'] / 'records' / (tid + '.json')
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding='utf-8'))
        text = (r.get('parsed') or {}).get('transcription') or ''
        if r.get('status') == 'ok' and text.strip():
            out[tid] = text
    return out


def build_jobs(limit=None, write_frozen=True):
    cases = inputs.load_cases(with_pass2=True)
    if limit:
        cases = cases[:int(limit)]
    p3 = load_pass3_texts([c['trace_id'] for c in cases])
    jobs, metas = [], {}
    for c in cases:
        job, meta = build_job(c, p3.get(c['trace_id']))
        jobs.append(job); metas[c['trace_id']] = meta
        if write_frozen:   # stage 2 reads the stage-1 prompt (to ignore echoed taxonomy text) and the pass-3 usage flags from here
            atomic(STAGE_DIRS['stage1'] / 'frozen_inputs' / 'jobs' / (c['trace_id'] + '.json'),
                   {'job': {**job, 'images': [r for r, _ in job['images']]}, 'meta': meta})
    return jobs, metas
