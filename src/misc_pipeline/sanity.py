"""Checks used by notebook 00: hardware, dataset layout, bundled resources, model templates, and a plumbing test with placeholder outputs."""
import hashlib, importlib, json, os, platform, shutil, subprocess, sys, tempfile
from pathlib import Path

from . import common
from .common import DATA, PRIVATE, RES, ROOT, expected_hashes, read_csv, sha256_file

# Rough weight sizes in GB (bf16 unless noted). Used only for the disk / GPU memory comparison; see README.
MODEL_GB = {'Qwen/Qwen3-VL-235B-A22B-Instruct': 475, 'Qwen/Qwen3.8-Flash-Next': 360, 'Qwen/Qwen3.8-27B': 56}
STAGE_MODELS = {'pass1': 'Qwen/Qwen3-VL-235B-A22B-Instruct', 'pass2': 'Qwen/Qwen3.8-Flash-Next', 'pass3': 'Qwen/Qwen3.8-Flash-Next',
                'stage1': 'Qwen/Qwen3.8-27B', 'stage2': 'Qwen/Qwen3.8-27B'}
REQUIRED_PY_MODULES = ['torch', 'transformers', 'vllm', 'huggingface_hub', 'PIL']


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ''


def gpus():
    out = _run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'])
    res = []
    for line in out.splitlines():
        name, mem = [x.strip() for x in line.rsplit(',', 1)]
        res.append({'name': name, 'gb': round(float(mem) / 1024, 1)})
    return res


def ram_gb():
    try:
        with open('/proc/meminfo') as f:
            return round(int(f.readline().split()[1]) / 1024 / 1024, 1)
    except Exception:
        return None


def hardware_report(cache_dir=None):
    """Prints the machine and what each stage needs. Returns {'ok': bool, 'rows': [...]}."""
    g = gpus(); total = sum(x['gb'] for x in g)
    cache = Path(cache_dir or os.environ.get('HF_HOME') or Path.home() / '.cache' / 'huggingface')
    cache.mkdir(parents=True, exist_ok=True)
    free = round(shutil.disk_usage(cache).free / 1e9)
    info = {'python': sys.version.split()[0], 'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'ram_gb': ram_gb(),
            'gpus': g, 'gpu_total_gb': total, 'disk_free_gb_at_model_cache': free, 'model_cache': str(cache)}
    try:
        import torch
        info.update(torch_version=torch.__version__, cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
    except Exception as exc:
        info['torch'] = 'not importable: %s' % exc
    print(json.dumps(info, indent=2))
    rows = []
    for stage, model in STAGE_MODELS.items():
        need = MODEL_GB[model] * 1.25   # weights plus KV cache / activations headroom
        rows.append({'stage': stage, 'model': model, 'weights_gb': MODEL_GB[model], 'gpu_gb_needed_est': round(need), 'gpu_ok': total >= need})
    for r in rows:
        print('%-7s %-36s weights ~%3d GB | GPU memory needed ~%3d GB | %s' % (r['stage'], r['model'], r['weights_gb'], r['gpu_gb_needed_est'], 'OK' if r['gpu_ok'] else 'NOT ENOUGH GPU MEMORY'))
    need_disk = sum(MODEL_GB.values())
    print('disk for all three models ~%d GB; free at the model cache: %d GB -> %s' % (need_disk, free, 'OK' if free >= need_disk else 'download models one stage at a time and delete the previous ones'))
    return {'ok': bool(g) and all(r['gpu_ok'] for r in rows), 'info': info, 'rows': rows}


def modules_report():
    ok = True
    for m in REQUIRED_PY_MODULES:
        try:
            mod = importlib.import_module(m)
            print('%-18s %s' % (m, getattr(mod, '__version__', 'ok')))
        except Exception as exc:
            ok = False; print('%-18s MISSING (%s)' % (m, exc))
    return ok


def dataset_report():
    """Checks that the competition data is where the scripts expect it."""
    need = ['train.csv', 'test.csv', 'unlabelled.csv', 'sample_submission.csv', 'taxonomy_v1.csv']
    problems = []
    for n in need:
        if not (DATA / n).is_file():
            problems.append('missing file: %s' % (DATA / n))
    for d in ('images/test', 'images/train', 'images/questions'):
        if not (DATA / d).is_dir():
            problems.append('missing folder: %s' % (DATA / d))
    if not problems:
        test = read_csv(DATA / 'test.csv'); train = read_csv(DATA / 'train.csv'); tax = read_csv(DATA / 'taxonomy_v1.csv'); sample = read_csv(DATA / 'sample_submission.csv')
        if len(test) != 3498: problems.append('test.csv has %d rows, expected 3498' % len(test))
        if len(train) != 1500: problems.append('train.csv has %d rows, expected 1500' % len(train))
        if len(tax) != 75: problems.append('taxonomy_v1.csv has %d rows, expected 75' % len(tax))
        if {r['trace_id'] for r in sample} != {r['trace_id'] for r in test}: problems.append('sample_submission.csv ids differ from test.csv')
        miss = [r['image_file'] for r in test if not (DATA / 'images' / 'test' / r['image_file']).is_file()]
        if miss: problems.append('%d test images missing, e.g. %s' % (len(miss), miss[:3]))
        qmiss = [r['question_image_file'] for r in test if r['question_image_file'] and not (DATA / 'images' / 'questions' / r['question_image_file']).is_file()]
        if qmiss: problems.append('%d question diagrams missing, e.g. %s' % (len(qmiss), qmiss[:3]))
        from .stage2 import TAX_IDS
        if [r['id'] for r in tax] != TAX_IDS: problems.append('taxonomy order differs from the expected one')
        print('test rows %d | train rows %d | taxonomy %d | test images %d | question diagrams used by test rows %d' %
              (len(test), len(train), len(tax), len(test) - len(miss), sum(1 for r in test if r['question_image_file'])))
    for p in problems:
        print('PROBLEM:', p)
    if not problems:
        print('dataset OK at', DATA)
    return not problems


def resources_report():
    """Bundled prompts, the knowledge-base files and the reference-working cache (the last two are private artifacts)."""
    exp = expected_hashes(); problems = []
    for f in sorted((RES / 'prompts').glob('*.txt')):
        print('prompt  %-40s %6d bytes' % (f.name, f.stat().st_size))
    rules = RES / 'stage2_rules.json'
    if not rules.is_file(): problems.append('missing ' + str(rules))
    corpus = RES / 'corpus' if (RES / 'corpus' / 'library.jsonl').is_file() else PRIVATE / 'corpus'
    for n, h in exp['stage2_expect']['corpus'].items():
        p = corpus / n
        if not p.is_file(): problems.append('missing knowledge-base file %s (see README: Private artifacts)' % p)
        elif sha256_file(p) != h: problems.append('%s differs from the file used for the submission' % p)
        else: print('corpus  %-20s sha256 OK' % n)
    ref = RES / 'reference' / 'question_solution_cache.json'
    if not ref.is_file(): ref = PRIVATE / 'reference' / 'question_solution_cache.json'
    if not ref.is_file(): problems.append('missing reference-working cache (see README: Private artifacts)')
    elif sha256_file(ref) != exp['reference_cache_sha256']: problems.append('reference cache differs from the one used for the submission')
    else: print('reference cache sha256 OK')
    for p in problems:
        print('PROBLEM:', p)
    return not problems


def template_report(stages=('pass1', 'pass2', 'stage1')):
    """Renders each model's chat template with the thinking settings from configs/models.json; confirms the options change the prompt."""
    from transformers import AutoTokenizer
    from .llm import load_specs
    specs = load_specs(); ok = True; seen = set()
    for st in stages:
        s = specs[st]
        if s['hf_id'] in seen:
            continue
        seen.add(s['hf_id'])
        try:
            tok = AutoTokenizer.from_pretrained(s['hf_id'])
            msgs = [{'role': 'user', 'content': 'hello'}]
            kw = s.get('chat_template_kwargs') or {}
            a = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            b = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **{**kw, 'enable_thinking': False}) if kw.get('enable_thinking') else a
            print('%s | kwargs %s | thinking switch changes the prompt: %s' % (s['hf_id'], kw, a != b if kw.get('enable_thinking') else 'n/a'))
            print('   prompt tail:', repr(a[-80:]))
            if kw.get('enable_thinking') and a == b:
                ok = False; print('   WARNING: enable_thinking has no effect on this template; adjust chat_template_kwargs in configs/models.json')
        except Exception as exc:
            ok = False; print('%s: could not load tokenizer/template: %s' % (s['hf_id'], exc))
    return ok


def plumbing_test(n=3):
    """Runs pass 1 on n test rows with the placeholder backend (no GPU, no weights) into a temporary folder."""
    from . import pass1
    from .runner import run_stage
    old = os.environ.get('MISC_BACKEND'); os.environ['MISC_BACKEND'] = 'mock'
    try:
        jobs = pass1.build_jobs(n)
        with tempfile.TemporaryDirectory() as tmp:
            s = run_stage('pass1', jobs, Path(tmp) / 'pass1', pass1.parse, label_images=False)
        print('plumbing test:', s)
        return s['complete']
    finally:
        if old is None: os.environ.pop('MISC_BACKEND', None)
        else: os.environ['MISC_BACKEND'] = old
