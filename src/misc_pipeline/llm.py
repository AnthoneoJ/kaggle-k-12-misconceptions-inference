"""Local model inference (vLLM, in-process). No network service is started or contacted.

Every stage hands `Engine.generate` a list of requests and gets one result per request:
    {'content', 'reasoning', 'finish_reason', 'prompt_tokens', 'completion_tokens'}
Set MISC_BACKEND=mock to exercise the pipeline plumbing without a GPU (outputs are placeholders).
"""
import base64, gc, json, mimetypes, os, re
from pathlib import Path

from .common import ROOT

THINK_CLOSE = '</think>'
THINK_OPEN = '<think>'


def load_specs():
    return json.loads((ROOT / 'configs' / 'models.json').read_text(encoding='utf-8'))


def split_thinking(text, thinking_enabled, finish_reason):
    """Split raw model text into (reasoning, content).

    Qwen thinking models write `<think> ... </think> answer` (the opening tag may already be part of the prompt).
    Unfinished thinking (no closing tag, cut off by the token limit) is all reasoning and leaves content empty.
    """
    text = text or ''
    if THINK_CLOSE in text:
        head, _, tail = text.rpartition(THINK_CLOSE)
        head = head.strip()
        if head.startswith(THINK_OPEN):
            head = head[len(THINK_OPEN):]
        return head.strip('\n'), tail.strip('\n')
    if thinking_enabled and finish_reason != 'stop':
        return text.replace(THINK_OPEN, '', 1).strip('\n'), ''
    return None, text.strip('\n')


def image_part(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or 'image/jpeg'
    return {'type': 'image_url', 'image_url': {'url': 'data:%s;base64,%s' % (mime, base64.b64encode(path.read_bytes()).decode())}}


def build_content(images, prompt, label_images=True):
    """images: [(role, path)]. Layout: optional 'Image: <role>' label, the image, then the prompt."""
    content = []
    for role, path in images:
        if label_images:
            content.append({'type': 'text', 'text': 'Image: ' + role})
        content.append(image_part(path))
    content.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': content}]


class VLLMBackend:
    def __init__(self, spec, runtime):
        import torch
        from vllm import LLM
        self.spec, self.runtime = spec, runtime
        gpus = torch.cuda.device_count()
        if gpus == 0:
            raise RuntimeError('No CUDA device visible; run notebook 00 for the hardware check.')
        kw = dict(model=spec['hf_id'], tensor_parallel_size=spec.get('tensor_parallel_size') or gpus,
                  max_model_len=spec['max_model_len'], gpu_memory_utilization=runtime.get('gpu_memory_utilization', 0.9),
                  limit_mm_per_prompt={'image': spec.get('max_images', 2)}, seed=runtime.get('seed', 0),
                  enable_prefix_caching=True)
        if spec.get('quantization'):
            kw['quantization'] = spec['quantization']
        kw.update(spec.get('vllm_extra', {}))
        self.llm = LLM(**kw)

    def generate(self, requests):
        from vllm import SamplingParams
        s = self.spec
        sp = SamplingParams(temperature=s['temperature'], top_p=s['top_p'], top_k=s['top_k'], max_tokens=s['max_tokens'])
        outs = self.llm.chat([r['messages'] for r in requests], sampling_params=sp, use_tqdm=True,
                             chat_template_kwargs=s.get('chat_template_kwargs') or None)
        thinking = bool((s.get('chat_template_kwargs') or {}).get('enable_thinking'))
        results = []
        for o in outs:
            c = o.outputs[0]
            reasoning, content = split_thinking(c.text, thinking, c.finish_reason)
            results.append({'content': content, 'reasoning': reasoning, 'finish_reason': c.finish_reason,
                            'prompt_tokens': len(o.prompt_token_ids or []), 'completion_tokens': len(c.token_ids)})
        return results

    def close(self):
        try:
            del self.llm
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
            destroy_model_parallel(); destroy_distributed_environment()
        except Exception as exc:   # best effort: restarting the kernel always frees the GPUs
            print('GPU memory release incomplete (restart the kernel before the next stage):', exc)


class MockBackend:
    """Placeholder outputs in each stage's expected format; for plumbing tests only."""
    def __init__(self, spec, runtime, stage):
        self.stage = stage

    def generate(self, requests):
        out = []
        for r in requests:
            if self.stage == 'pass1':
                c, reason = 'x + 1 = 2', None
            elif self.stage in ('pass2', 'pass3'):
                c, reason = 'FINAL_TRANSCRIPTION\nx + 1 = 2\nREVIEW_NOTES\nmock review notes', 'mock reasoning'
            elif self.stage == 'stage1':
                c = ('ANALYSIS: mock\n```json\n{"candidates": [{"code": "NONE", "why": "mock", "depends_on_reading": ""}, '
                     '{"code": "OTH-00", "why": "mock", "depends_on_reading": ""}, {"code": "ALG-01", "why": "mock", "depends_on_reading": ""}], '
                     '"search_queries": ["mock query"]}\n```')
                reason = 'mock reasoning mentioning ALG-02 and INT-04.'
            else:
                c, reason = '```json\n{"ranked_codes": ["NONE", "OTH-00", "ALG-01"]}\n```', 'mock reasoning'
            out.append({'content': c, 'reasoning': reason, 'finish_reason': 'stop', 'prompt_tokens': 0, 'completion_tokens': 0})
        return out

    def close(self):
        pass


class Engine:
    def __init__(self, stage):
        specs = load_specs()
        self.stage, self.spec, self.runtime = stage, specs[stage], specs['runtime']
        if os.environ.get('MISC_BACKEND', 'vllm') == 'mock':
            self.backend = MockBackend(self.spec, self.runtime, stage)
        else:
            self.backend = VLLMBackend(self.spec, self.runtime)

    def generate(self, requests):
        return self.backend.generate(requests)

    def close(self):
        self.backend.close()
