"""Resumable batch runner: one JSON record per row under <out>/records/<trace_id>.json."""
import json, time
from pathlib import Path

from .common import atomic, digest
from .llm import Engine, build_content, load_specs


def job_signature(stage, spec, job):
    return digest({'stage': stage, 'spec': spec, 'prompt': job['prompt'],
                   'images': [[role, Path(p).name] for role, p in job['images']]})


def load_records(out_dir):
    recs = {}
    d = Path(out_dir) / 'records'
    if d.is_dir():
        for p in d.glob('*.json'):
            try:
                r = json.loads(p.read_text(encoding='utf-8'))
            except ValueError:
                continue
            recs[r['trace_id']] = r
    return recs


def make_record(stage, job, sig, spec, res, parse, seconds):
    rec = {'trace_id': job['trace_id'], 'signature': sig, 'status': 'error', 'stage': stage, 'model': spec['hf_id'], 'provider': 'local',
           'finish_reason': res['finish_reason'], 'content': res['content'], 'seconds': seconds,
           'usage': {'prompt_tokens': res['prompt_tokens'], 'completion_tokens': res['completion_tokens']}}
    if spec.get('keep_reasoning'):
        rec['reasoning'] = res['reasoning']
    try:
        if res['finish_reason'] != 'stop':
            raise ValueError('Incomplete response: ' + str(res['finish_reason']))
        rec['parsed'] = parse({'content': res['content'], 'reasoning': res['reasoning']})
        rec['status'] = 'ok'
    except Exception as exc:   # a bad answer is saved and retried in the next round
        rec['error'] = '%s: %s' % (type(exc).__name__, exc)
        if not spec.get('keep_reasoning') and res.get('reasoning'):
            rec['reasoning_tail'] = res['reasoning'][-2000:]
    return rec


def run_stage(stage, jobs, out_dir, parse, label_images=True, engine=None):
    """Run every job without an ok record (same signature), in chunks, with retry rounds. Returns a summary dict."""
    specs = load_specs(); spec, rt = specs[stage], specs['runtime']
    out_dir = Path(out_dir)
    sigs = {j['trace_id']: job_signature(stage, spec, j) for j in jobs}
    have = load_records(out_dir)
    todo = [j for j in jobs if not (j['trace_id'] in have and have[j['trace_id']]['status'] == 'ok' and have[j['trace_id']]['signature'] == sigs[j['trace_id']])]
    print('%s: %d rows, %d already done, %d to run' % (stage, len(jobs), len(jobs) - len(todo), len(todo)), flush=True)
    own = engine is None
    if todo and own:
        engine = Engine(stage)
    t0 = time.monotonic(); n_ok = n_err = 0
    try:
        for rnd in range(rt.get('retry_rounds', 2) + 1):
            if not todo:
                break
            if rnd:
                print('retry round %d: %d rows' % (rnd, len(todo)), flush=True)
            failed = []
            for i in range(0, len(todo), rt.get('chunk_size', 96)):
                chunk = todo[i:i + rt.get('chunk_size', 96)]
                requests = [{'key': j['trace_id'], 'messages': build_content(j['images'], j['prompt'], label_images)} for j in chunk]
                started = time.monotonic()
                results = engine.generate(requests)
                per = (time.monotonic() - started) / max(1, len(chunk))
                for j, res in zip(chunk, results):
                    rec = make_record(stage, j, sigs[j['trace_id']], spec, res, parse, per)
                    atomic(out_dir / 'records' / (j['trace_id'] + '.json'), rec)
                    if rec['status'] == 'ok':
                        n_ok += 1
                    else:
                        n_err += 1; failed.append(j)
                print('[%s] %d ok, %d failed attempts | %.1f rows/min' % (time.strftime('%H:%M:%S'), n_ok, n_err, n_ok / max(time.monotonic() - t0, 1) * 60), flush=True)
            todo = failed
    finally:
        if own and engine is not None:
            engine.close()
    have = load_records(out_dir)
    ok = [j['trace_id'] for j in jobs if j['trace_id'] in have and have[j['trace_id']]['status'] == 'ok' and have[j['trace_id']]['signature'] == sigs[j['trace_id']]]
    summary = {'stage': stage, 'model': spec['hf_id'], 'expected': len(jobs), 'completed': len(ok), 'complete': len(ok) == len(jobs),
               'failed_rows': [j['trace_id'] for j in jobs if j['trace_id'] not in set(ok)][:50], 'elapsed_s': round(time.monotonic() - t0)}
    atomic(out_dir / 'summary.json', summary)
    return summary
