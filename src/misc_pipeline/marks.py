"""Pass-3 mark helpers: {{likely reading / pass-2 reading}} marks and the section splitter."""
import re


def split_sections(text):
    if not isinstance(text, str) or not text.strip(): raise ValueError('Empty final response')
    a = re.search(r'^[ \t#*]*FINAL_TRANSCRIPTION[ \t:*]*$', text, re.M)
    b = re.search(r'^[ \t#*]*REVIEW_NOTES[ \t:*]*$', text, re.M)
    if not a or not b or a.end() > b.start(): raise ValueError('FINAL_TRANSCRIPTION / REVIEW_NOTES sections not found')
    return {'transcription': text[a.end():b.start()].strip('\n'), 'review_notes': text[b.end():].strip('\n')}

# {{ correction / pass-2 reading }}; 'blank' means nothing. Marks without ' / ' (for example {{unclear: ...}}) are left as they are.
MARK = re.compile(r'\{\{\s*(.*?)\s*\}\}', re.S)
def _alt(m, i):
    parts = [p.strip() for p in re.split(r'\s+/\s+', m.group(1))]
    if len(parts) < 2: return m.group(0)
    v = parts[i] if i < len(parts) else parts[-1]
    return '' if v.lower() == 'blank' else v
def collapse(t, i): return MARK.sub(lambda m: _alt(m, i), t)       # i=0: first reading, i=1: second reading
def norm(t): return re.sub(r'\s+', ' ', t).strip()
def new_marks(p3, p2):
    old = {norm(x) for x in MARK.findall(p2)}
    return [norm(x) for x in MARK.findall(p3) if norm(x) not in old]
