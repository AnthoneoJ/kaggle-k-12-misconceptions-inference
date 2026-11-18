"""Stage 2: audit of the stage-1 candidates. Retrieval, prompt assembly and parsing."""
import hashlib, json, math, re
from collections import Counter
from pathlib import Path

from . import inputs
from .common import RES, PRIVATE, STAGE_DIRS, check_hash, digest, expected_hashes, prompt as load_prompt, taxonomy, test_rows

TAX_IDS = ['PV-01', 'OPS-01', 'OPS-02', 'OOO-01', 'OOO-02', 'KPF-01', 'KPF-02', 'KPF-03', 'KPF-04', 'INT-01', 'INT-02', 'INT-03', 'INT-04', 'FRA-01', 'FRA-02', 'FRA-03', 'FRA-04', 'FRA-05', 'FRA-06', 'Dec-01', 'Dec-02', 'EXP-01', 'EXP-02', 'EXP-03', 'EXP-07', 'RAT-01', 'RAT-02', 'ALG-01', 'ALG-02', 'ALG-03', 'ALG-04', 'ALG-05', 'ALG-06', 'MEA-01', 'MEA-02', 'MEA-03', 'MEA-04', 'MEA-05', 'GEO-01', 'ANG-01', 'ANG-02', 'COR-01', 'STA-01', 'STA-03', 'STA-04', 'STA-05', 'PRO-01', 'PRO-02', 'PRO-03', 'PRO-04', 'FUN-01', 'INV-01', 'LIN-01', 'SIM-01', 'QUA-02', 'LOG-02', 'EXP-04', 'EXP-05', 'EXP-06', 'SEQ-01', 'SEQ-02', 'SET-01', 'ANS-01', 'ANS-02', 'CAL-01', 'VIS-01', 'WOP-01', 'WOP-02', 'FRM-01', 'NOT-01', 'NOT-02', 'CHK-01', 'UNT-01', 'REL-01', 'OTH-00']
INCLUDE_PASS2_NOTES = True
SILENT_EDIT_POLICY = 'fallback_pass2'
RETRIEVAL = {'partners_per_candidate': 2, 'examples_per_candidate': 2, 'root_bonus': 2.0, 'max_precedents': 4, 'precedent_chars': 900}
CODES = set(TAX_IDS) | {'NONE'}
AUDIT_TEMPLATE_PATH = RES / 'prompts' / 'stage2_audit_template_v6.txt'
EXPECT = expected_hashes()['stage2_expect']
check_hash('stage2 audit template', hashlib.sha256(AUDIT_TEMPLATE_PATH.read_bytes()).hexdigest(), EXPECT['audit_template'])


# ---- retrieval and prompt builders ----
def mentioned(text, codes):
    # Every taxonomy code / NONE named in free text, in first-mention order. Includes codes the model rejected.
    if not text:
        return []
    pattern = re.compile(r'(?<![A-Za-z0-9-])(' + '|'.join(re.escape(c) for c in sorted(codes, key=len, reverse=True)) + r')(?![A-Za-z0-9-])')
    return list(dict.fromkeys(pattern.findall(text)))




def tax_order(codes, taxonomy_ids):
    pos = {c: i for i, c in enumerate(taxonomy_ids + ['NONE'])}
    return sorted(set(codes), key=lambda c: pos.get(c, 10 ** 6))


def _norm(text):
    return ' '.join(re.sub(r'[*_`"“”\'‘’]', ' ', text or '').split()).lower()


def considered_mentions(reasoning, codes, stage1_prompt):
    # First mention of each code that is not a quotation of the stage-1 prompt (taxonomy, conventions, examples).
    # Codes that only ever appear inside such quotations are dropped.
    prompt_norm = _norm(stage1_prompt)
    out = {}
    for code in mentioned(reasoning, codes):
        for m in re.finditer(r'(?<![A-Za-z0-9-])' + re.escape(code) + r'(?![A-Za-z0-9-])', reasoning):
            start = max(reasoning.rfind('\n', 0, m.start()), reasoning.rfind('. ', 0, m.start()) + 1, 0)
            ends = [e for e in (reasoning.find('\n', m.end()), reasoning.find('. ', m.end())) if e != -1]
            sentence = reasoning[start:(min(ends) + 1 if ends else len(reasoning))]
            words = _norm(sentence).split()
            echoed = len(words) >= 6 and any(' '.join(words[i:i + 6]) in prompt_norm for i in range(0, len(words) - 5))
            if not echoed:
                snippet = ' '.join(sentence.split())
                out[code] = snippet if len(snippet) <= 320 else snippet[:320] + '…'
                break
    return out




TOKEN = re.compile(r'[a-z0-9]+')


def tokens(text):
    return TOKEN.findall((text or '').lower())


class BM25:
    def __init__(self, docs, k1=1.4, b=0.75):
        self.docs = [Counter(tokens(d)) for d in docs]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def score(self, query, i):
        d, s = self.docs[i], 0.0
        for t in set(tokens(query)):
            if t in d:
                f = d[t]
                s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class Retriever:
    def __init__(self, library, cards, partners, taxonomy_ids, settings):
        self.lib = [r for r in library if not r['exclude_from_teaching']]
        self.findings = [r for r in self.lib if r['kind'] == 'finding']
        self.reviews = [r for r in self.lib if r['kind'] == 'review']
        self.f_bm = BM25([(r.get('span') or '') + ' ' + r['text'] + ' ' + (r.get('topic') or '') for r in self.findings])
        self.r_bm = BM25([r['text'] for r in self.reviews])
        self.cards, self.partners, self.tax, self.cfg = cards, partners, taxonomy_ids, settings

    def card_codes(self, s1):
        listed = set(s1['listed'])
        chosen = set(listed)
        for c in listed:
            chosen |= {p['code'] for p in self.partners.get(c, [])[:self.cfg['partners_per_candidate']]}
        partner_pool = {p['code'] for c in listed for p in self.partners.get(c, [])}
        chosen |= {c for c in s1['also'] if c in partner_pool}
        return tax_order(chosen, self.tax)

    def examples(self, tid, code, query):
        if code == 'NONE':
            pool = [i for i, r in enumerate(self.findings) if r['row_root_code'] == 'NONE' and r['trace_id'] != tid]
        else:
            pool = [i for i, r in enumerate(self.findings) if r['code'] == code and r['trace_id'] != tid]
        ranked = sorted(pool, key=lambda i: (self.f_bm.score(query, i) + (self.cfg['root_bonus'] if self.findings[i]['is_root'] else 0)), reverse=True)
        return [self.findings[i] for i in ranked[:self.cfg['examples_per_candidate']]]

    def precedents(self, tid, in_play, query):
        pool = [i for i, r in enumerate(self.reviews)
                if r['trace_id'] != tid and len(r.get('pair') or []) == 2 and r['pair'][0] != r['pair'][1]
                and set(r['pair']) <= in_play]
        ranked = sorted(pool, key=lambda i: self.r_bm.score(query, i), reverse=True)
        seen, out = set(), []
        for i in ranked:
            r = self.reviews[i]
            if r['trace_id'] in seen:
                continue
            seen.add(r['trace_id']); out.append(r)
            if len(out) >= self.cfg['max_precedents']:
                break
        return out


def render_stage2(template, case, s1, retriever, taxonomy_ids):
    query = ' '.join(list(s1['why'].values()) + s1['queries'])
    listed = tax_order(s1['listed'], taxonomy_ids)
    cand_text = '\n'.join('- ' + c + ': ' + (s1['why'].get(c) or '') + (' [depends on reading: ' + s1['depends'][c] + ']' if s1['depends'].get(c) else '') for c in listed)
    also = tax_order(s1['also'], taxonomy_ids)
    also_text = '\n'.join('- ' + c + ': "' + s1['also'][c] + '"' for c in also) or '(none)'
    card_codes = retriever.card_codes(s1)
    cards_text = '\n\n'.join(retriever.cards[c]['text'] for c in card_codes if c in retriever.cards)
    ex_lines, ex_ids = [], []
    for c in listed:
        for r in retriever.examples(case['trace_id'], c, query):
            label = (r['code'] + (' (root)' if r['is_root'] else ' (not the root; row root ' + r['row_root_code'] + ')')) if c != 'NONE' else 'NONE row (minor finding ' + str(r['code']) + ')'
            span = (' Writing: "' + ' '.join(r['span'].split())[:160] + '".') if r.get('span') else ''
            ex_lines.append('- ' + label + ' — topic ' + (r.get('topic') or '') + '.' + span + ' ' + ' '.join(r['text'].split())[:420])
            ex_ids.append(r['id'])
    in_play = set(card_codes) | set(listed)
    precs = retriever.precedents(case['trace_id'], in_play, query)
    prec_text = '\n\n'.join('- ' + (' vs '.join(r['pair'])) + ' — ' + ' '.join(r['text'].split())[:retriever.cfg['precedent_chars']] for r in precs) or '(none)'
    sections = {'candidates': cand_text, 'also_considered': also_text, 'rule_cards': cards_text,
                'annotated_examples': '\n'.join(ex_lines) or '(none)', 'review_precedents': prec_text}
    prompt = template.format(question=case['question'], answer=case['answer'],
        diagram_note='The image labelled student_working is the original student photo. ' + (
            'The separately labelled question_diagram is the supplied QUESTION diagram.' if case['question_asset'] else 'There is no separate question diagram.'),
        reference_working=case['reference_working'], reference_caveats=case['reference_caveats'],
        pass1=case['pass1'], pass2=case['pass2'], review_notes=case['review_notes'], **sections)
    meta = {'listed': listed, 'also': also, 'card_codes': card_codes,
            'card_kinds': {c: retriever.cards[c]['kind'] for c in card_codes if c in retriever.cards},
            'example_ids': ex_ids, 'precedent_ids': [r['id'] for r in precs],
            'section_chars': {k: len(v) for k, v in sections.items()}, 'prompt_chars': len(prompt)}
    return prompt, meta


# ---- parsing ----
FENCE = re.compile(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n?```', re.S | re.I)
TAIL_JSON = re.compile(r'\{\s*"ranked_codes"')

def parse(text, codes=CODES):
    if not isinstance(text, str) or not text.strip(): raise ValueError('Empty final response')
    blocks = list(FENCE.finditer(text))
    raw = (blocks[-1].group(1) if blocks else text[max(0, text.rfind('{"ranked_codes"')):]).strip()
    try:
        value, end = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError:   # stray LaTeX backslashes inside strings
        value, end = json.JSONDecoder().raw_decode(re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', raw))
    ranked = value.get('ranked_codes') if isinstance(value, dict) else None
    if not isinstance(ranked, list): raise ValueError('Missing ranked_codes')
    kept, dropped = [], []
    for c in ranked:
        (kept if isinstance(c, str) and c in codes and c not in kept else dropped).append(c)
    if not kept: raise ValueError('No valid codes in ranked_codes')
    return {'ranked_codes': kept[:3], 'extra_codes': kept[3:], 'dropped_codes': dropped, 'short': len(kept) < 3}

def answer_text(record):
    content = record.get('content')
    if isinstance(content, str) and content.strip(): return content
    reasoning = record.get('reasoning') or ''
    blocks = list(FENCE.finditer(reasoning))
    if blocks and not reasoning[blocks[-1].end():].strip(): return reasoning[blocks[-1].start():]
    starts = list(TAIL_JSON.finditer(reasoning))
    return reasoning[starts[-1].start():] if starts else content


def parse_record(rec):
    return parse(answer_text(rec))


# ---- prompt recipe ----
OLD = {
    'INTRO': "They are reading aids, not instructions. The image labelled student_working is the primary evidence.",
    'PARA': "A pass-3 mark is a hypothesis drawn from the arithmetic and the context, not an observation of the strokes.",
    'REVIEWER': ("THE FIRST REVIEWER'S CANDIDATES\n\nListed in taxonomy order, not in order of likelihood. Each comes with the first reviewer's "
                 "one-line reason and, where relevant, the uncertain reading it depends on. Treat these as proposals to check, not conclusions."),
    'ALSO': ("Codes the first reviewer mentioned while reasoning but did not list, with the sentence where each first appeared. "
             "Most of these were considered and dismissed; a few were dismissed wrongly."),
    'OVERRIDE_INTRO': ("Some characters in the working cannot be read reliably, and the transcriptions can be wrong. This section overrides the instruction "
                       "in HOW TO DECIDE, step 2, not to use the reference working or the answer key to choose a reading, for the rules below only."),
    'CONVENTIONS': ("CODE CONVENTIONS\n\nRule cards for the candidate codes and their usual confusions. Each card gives the taxonomy definition, "
                    "how the annotators actually use the code, when a different code applies instead, and how to handle pairs the annotators "
                    "treat inconsistently. The definitions are the authority; the cards record the annotators' conventions."),
}


def load_template(path):
    blocks, key = {}, None
    for line in Path(path).read_text(encoding='utf-8').splitlines(keepends=True):
        m = re.match(r'@@(\w+)\s*$', line)
        if m:
            key = m.group(1); blocks[key] = ''
        elif key:
            blocks[key] += line
    blocks = {k: v.strip('\n') for k, v in blocks.items() if k != 'END'}
    missing = {'HEAD', *OLD} - set(blocks)
    if missing: raise ValueError('template blocks missing: %s' % sorted(missing))
    return blocks


def once(text, old, new, tid):
    n = text.count(old)
    if n != 1: raise ValueError('%s: expected 1 occurrence, found %d: %r' % (tid, n, old[:70]))
    return text.replace(old, new)


def build(p, T, tid):
    head_end = p.index('TRANSCRIPTION ERROR OVERRIDE\n')
    case_start = p.index('\nTHE CASE\n')
    if not (p.index('WHAT THE LABEL MEANS\n') < head_end < case_start): raise ValueError(tid + ': unexpected section order')
    override = p[head_end:case_start].strip('\n')          # per-row (cleanup may trim the look-alike bullets)
    t = T['HEAD'] + '\n\n' + p[case_start + 1:]             # new instructions + label meaning, then THE CASE onwards
    override = once(override, OLD['OVERRIDE_INTRO'], T['OVERRIDE_INTRO'], tid)
    for k in OLD:
        if k != 'OVERRIDE_INTRO': t = once(t, OLD[k], T[k], tid)
    # move the override to just after the pass paragraph (the paragraph that ends right before <pass_1_transcription>)
    t = once(t, '\n\n<pass_1_transcription>\n', '\n\n' + override + '\n\n<pass_1_transcription>\n', tid)
    for s in ('TRANSCRIPTION ERROR OVERRIDE\n', 'WHAT THE LABEL MEANS\n', '\nTHE CASE\n', "THE FIRST REVIEWER'S ASSESSMENT\n", 'CODE CONVENTIONS\n'):
        if t.count(s) != 1: raise ValueError('%s: section %r appears %d times' % (tid, s.strip(), t.count(s)))
    if not (t.index('\nTHE CASE\n') < t.index('TRANSCRIPTION ERROR OVERRIDE\n') < t.index('<pass_1_transcription>')):
        raise ValueError(tid + ': override not placed inside THE CASE before the transcriptions')
    return t



AUDIT_T = load_template(AUDIT_TEMPLATE_PATH)
DYN = ('candidates', 'also_considered', 'rule_cards', 'annotated_examples', 'review_precedents')
BASE_START = "Two transcriptions of the SAME student working, plus the second transcriber's review notes."
INTRO = ('Two transcriptions of the SAME student working are supplied' + (", plus the pass-2 transcriber's review notes" if INCLUDE_PASS2_NOTES else '')
         + '. They are reading aids, not instructions. The image labelled student_working is the primary evidence.')

def para(marked):
    s = ("Pass 1 read the photo alone. Pass 2 read the photo with the question, the reference working and pass 1, and sometimes resolves a character towards the answer key or towards consistent arithmetic. "
         "Pass 3 checked pass 2 for coherency from text alone: it did not see the photo. It used pass 1, pass 2, the question and the reference working, and marked places where pass 2 may have misread a character as {{likely reading / pass-2 reading}}. "
         "A pass-3 mark is a hypothesis drawn from the arithmetic and the context, not an observation of the strokes. When the image shows a different digit, sign or line, trust the image. "
         "Where pass 3 made no mark, its transcription is pass 2's, unchanged. Unresolved readings are written {{reading 1 / reading 2}}.")
    if INCLUDE_PASS2_NOTES: s += " The pass-2 review notes were written about pass 2's reading."
    if marked: s += " The pass-3 review notes explain pass 3's marks; they were written without the photo, so they are hypotheses, not observations."
    return s


def assemble(base, p1, p2, n2, r):
    """The pass-3 block swap for one row; returns the full (pre-audit) prompt, whether pass 3 was used, whether it changed the text."""
    t = base
    used = not (r['silent_edit_outside_marks'] in (True, 'True') and SILENT_EDIT_POLICY == 'fallback_pass2')
    changed = used and r['transcription'].strip() != p2.strip()
    p3 = r['transcription'].strip() if changed else p2
    a = t.index(BASE_START); e = t.index('</pass_2_review_notes>') + len('</pass_2_review_notes>')
    region = (INTRO + '\n\n' + para(changed) + '\n\n<pass_1_transcription>\n' + p1 + '\n</pass_1_transcription>\n\n<pass_3_transcription>\n' + p3 + '\n</pass_3_transcription>')
    if INCLUDE_PASS2_NOTES: region += '\n\n<pass_2_review_notes>\n' + n2 + '\n</pass_2_review_notes>'
    if changed: region += '\n\n<pass_3_review_notes>\n' + r['review_notes'].strip() + '\n</pass_3_review_notes>'
    t = t[:a] + region + t[e:]
    assert '<pass_2_transcription>' not in t and t.count('<pass_3_transcription>') == 1
    return t, bool(used), bool(changed)


# ---- knowledge base, jobs ----
MID = load_prompt('stage2_mid.txt')
STATIC_PREFIX = load_prompt('stage2_static_prefix.txt')
check_hash('stage2 MID', hashlib.sha256(MID.encode()).hexdigest(), EXPECT['mid'])
check_hash('stage2 static prefix', hashlib.sha256(STATIC_PREFIX.encode()).hexdigest(), EXPECT['static_prefix'])


def load_kb():
    corpus = RES / 'corpus'
    if not (corpus / 'library.jsonl').is_file():
        corpus = PRIVATE / 'corpus'
    sha = {n: hashlib.sha256((corpus / n).read_bytes()).hexdigest() for n in ('library.jsonl', 'cards.json', 'partners.json')}
    if sha != EXPECT['corpus']:
        raise ValueError('corpus files differ from the ones used for the submission: ' + str(sha))
    lib = [json.loads(l) for l in (corpus / 'library.jsonl').read_text(encoding='utf-8').splitlines()]
    cards = json.loads((corpus / 'cards.json').read_text(encoding='utf-8'))['cards']
    partners = json.loads((corpus / 'partners.json').read_text(encoding='utf-8'))['partners']
    rules = json.loads((RES / 'stage2_rules.json').read_text(encoding='utf-8'))
    for c, t in rules['CARD_APPEND'].items():
        assert c in cards, c
        cards[c] = dict(cards[c], text=cards[c]['text'].rstrip() + '\n\n' + t)
    for c, p in rules['PARTNER_FRONT'].items():
        front = {x for x, _ in p}
        partners[c] = [{'code': x, 'count': n} for x, n in p] + [q for q in partners.get(c, []) if q['code'] not in front]
    return Retriever(lib, cards, partners, TAX_IDS, RETRIEVAL)


def s1_inputs(tid, s1_dir):
    rec = json.loads((s1_dir / 'records' / (tid + '.json')).read_text(encoding='utf-8'))
    s1job = json.loads((s1_dir / 'frozen_inputs' / 'jobs' / (tid + '.json')).read_text(encoding='utf-8'))
    cands = rec['parsed']['candidates']; listed = [c['code'] for c in cands]
    ctx = considered_mentions(rec.get('reasoning') or '', TAX_IDS + ['NONE'], s1job['job']['prompt'])
    return ({'listed': listed, 'why': {c['code']: c.get('why', '') for c in cands}, 'depends': {c['code']: c.get('depends_on_reading', '') for c in cands},
             'queries': rec['parsed']['search_queries'], 'also': {c: v for c, v in ctx.items() if c not in listed}}, s1job['meta'])


def stage1_ok_ids(s1_dir, rows):
    ok = set()
    for tid in rows:
        p = s1_dir / 'records' / (tid + '.json')
        if p.exists():
            r = json.loads(p.read_text(encoding='utf-8'))
            if r.get('status') == 'ok' and (r.get('parsed') or {}).get('candidates'):
                ok.add(tid)
    return ok


def build_jobs(limit=None):
    """Returns (jobs, metas). Rows without an ok stage-1 record are skipped (the submission builder falls back for them)."""
    s1_dir = STAGE_DIRS['stage1']
    cases = inputs.load_cases(with_pass2=True)
    if limit:
        cases = cases[:int(limit)]
    P3 = json.loads((STAGE_DIRS['pass3'] / 'pass3_transcriptions.json').read_text(encoding='utf-8'))
    ok = stage1_ok_ids(s1_dir, [c['trace_id'] for c in cases])
    ret = load_kb()
    jobs, metas = [], {}
    for case in cases:
        tid = case['trace_id']
        if tid not in ok:
            continue
        s1, s1meta = s1_inputs(tid, s1_dir)
        p, meta = render_stage2(MID, case, s1, ret, TAX_IDS)
        t, used, changed = assemble(STATIC_PREFIX + p, case['pass1'], case['pass2'], case['review_notes'], P3[tid])
        if s1meta.get('pass3_present') and (used != s1meta['used_pass3'] or changed != s1meta['changed']):
            raise RuntimeError('stage-1 job and the stage-2 rebuild disagree on the pass-3 text for ' + tid)
        audit = build(t, AUDIT_T, tid)
        images = [('student_working', case['student_image'])] + ([('question_diagram', case['question_image'])] if case['question_image'] else [])
        jobs.append({'trace_id': tid, 'prompt': audit, 'images': images})
        metas[tid] = {'listed': meta['listed'], 'also': meta['also'], 'card_codes': meta['card_codes'], 'example_ids': meta['example_ids'],
                      'precedent_ids': meta['precedent_ids'], 'used_pass3': used, 'changed': changed, 'prompt_chars': len(audit)}
    return jobs, metas
