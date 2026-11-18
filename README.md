# Misconception detection in K-12 student math workings

Five-stage inference pipeline (image transcription, review, coherency check, candidate codes, audit and ranking) that writes the competition submission file. It runs open-weight models locally with vLLM. No external service is called. Licence: Apache 2.0 (`LICENSE`).

## Pipeline

| Notebook | Stage | Model (`configs/models.json`) |
|---|---|---|
| `00_setup_and_sanity` | install modules, download data, check hardware, data, resources and chat templates | - |
| `01_pass1_transcribe` | transcribe the student photo | Qwen/Qwen3-VL-235B-A22B-Instruct |
| `02_pass2_review` | review the transcription against a reference working | Qwen/Qwen3.8-Flash-Next |
| `03_pass3_coherency` | mark doubtful readings as `{{a / b}}` | Qwen/Qwen3.8-Flash-Next |
| `04_stage1_candidates` | candidate misconception codes | Qwen/Qwen3.8-27B |
| `05_stage2_audit` | audit with retrieved rule cards and examples, rank top 3 | Qwen/Qwen3.8-27B |
| `06_make_submission` | write `submission/submission.csv` | - |
| `07_apply_jitter_rule` | optional: apply a post-processing rule to the stage-2 ranking, write `submission/submission_jitter.csv` | - |

## Requirements

- Linux, Python 3.10-3.12, CUDA 12.x, a vLLM build that supports Qwen3-VL and Qwen3.8. `pip install -r requirements.txt` (notebook 00 does it).
- GPUs (estimates from weight sizes, not measured): pass 1 needs 8 x 80 GB (or an FP8 build on 4 x 80 GB); passes 2-3 need 8 x 80 GB (or FP8); stages 1-2 need 1 x 80 GB. About 1 TB of disk for all weights. A cloud GPU platform such as RunPod works: an 8 x 80 GB pod for passes 1-3 and a single 80 GB pod for stages 1-2, with a persistent volume for the model cache.
- A Hugging Face token (and accepted dataset terms) or Kaggle credentials; see "Access tokens".
- Notebook 00 reports which of these your machine meets.

## Data

Notebook 00 downloads the competition data from Hugging Face (`colearnlab/misconception-detection`) or Kaggle (`misconception-detection-in-k-12-student-math-workings`). Expected layout:

```
data/competition_data/{train.csv, test.csv, sample_submission.csv, taxonomy_v1.csv, images/...}
```

The data may not be redistributed (`data/` is gitignored). Override paths with `MISC_DATA` (data) and `MISC_WORK` (stage outputs, default `work/`).

## Access tokens

Set these before starting Jupyter (or on the RunPod pod's environment), never inside a notebook that you commit:

| Needed for | What to set |
|---|---|
| Hugging Face dataset (gated) | Accept the terms on https://huggingface.co/datasets/colearnlab/misconception-detection, create a read token in your Hugging Face account settings, and `export HF_TOKEN=...`. Without it notebook 00 asks for the token interactively. |
| Model weights (about 900 GB in total) | The same `HF_TOKEN`. If a model page shows a licence or access prompt, accept it with the same account. A token also avoids anonymous download limits. |
| Kaggle download (alternative to Hugging Face) | Join the competition and accept its rules, create a Kaggle credential in your Kaggle account settings, and either save it as `~/.kaggle/kaggle.json` (`chmod 600`) or set `KAGGLE_USERNAME` and `KAGGLE_KEY`. Check Kaggle's current instructions if your CLI version expects a different format. |

Tokens are read from the environment only and are never written to the repository (`.env` is gitignored).

## Required private resources

Stage 2 and pass 2 need the knowledge base and the reference workings, which are derived from the competition training data and are not in a public repository. See `PRIVATE_ARTIFACTS.md` for the files and where to put them. Notebook 00 verifies them by sha256 (`resources/expected_hashes.json`).

## Running

1. Run `notebooks/00_setup_and_sanity.ipynb` and fix anything it flags.
2. Run notebooks 01 to 06 in order (07 is optional). **Each notebook runs a 5-row smoke test by default** (`LIMIT = 5` in the second cell). Set `LIMIT = None` for all 3,498 test rows; rows already finished are kept and skipped, so the full run resumes from the smoke test.
3. Each notebook ends with cells that free the GPUs and, optionally, delete the downloaded model from the cache (`DELETE_MODEL = True`).

Command line equivalent: `python scripts/run_pipeline.py all` (add `--limit 5` for a smoke test, or name stages: `python scripts/run_pipeline.py stage2 submission`). `MISC_BACKEND=mock` uses placeholder outputs, without GPUs, to test the code path only.

After a smoke test, notebook 06 fills the rows without model output with `NONE`; the file is only meaningful after the full run.

## Settings and reproducibility

- Passes 1-3 use temperature 0. Stages 1-2 are sampled (temperature 1.0, top-p 0.95, top-k 20, seed 0). vLLM batching is not bit-deterministic, so reruns differ in some rows.
- Thinking is switched on through the chat-template keywords in `configs/models.json` (`enable_thinking`, `reasoning_effort`). Notebook 00 prints the rendered templates; if a template ignores a keyword, edit it there.
- If a call fails, it is retried with identical settings (`retry_rounds`), then with the optional recovery variants of that stage; see "If rows fail". A row that still fails is left without a record, and the submission builder falls back (stage 2, then stage 1, then `NONE`).
- Every stage writes one JSON record per row under `work/<stage>/records`, so an interrupted run resumes where it stopped.

## If rows fail

Which rows fail depends on the hardware and the library versions, so the pipeline retries by itself and lets you change the settings for the failed rows only.

1. **Find them.** `work/<stage>/summary.json` lists `failed_rows`. The record `work/<stage>/records/<trace_id>.json` of a failed row has an `error` field.
2. **Automatic retries.** A failed row is retried with identical settings (`retry_rounds` in `configs/models.json`). Rows that still fail are then tried with each entry of that stage's `recovery` list, in order, and the first answer that parses is kept. A row saved this way has a `recovery_variant` field in its record, and `summary.json` counts them under `recovered_with_variant`.
3. **Change the recovery settings.** In `configs/models.json`, add or edit variants for a stage. `overrides` may set `temperature`, `top_p`, `top_k`, `max_tokens`, `repetition_penalty`, `min_p`, `presence_penalty` and `frequency_penalty`; everything else stays as configured for the stage.

```json
"stage2": { "...": "...", "recovery": [
  {"name": "warmer", "overrides": {"temperature": 0.7, "top_p": 0.8}},
  {"name": "short", "overrides": {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.1, "max_tokens": 4000}} ] }
```

   Pass 1 ships with two variants. Pass 1 runs at temperature 0, so a row that loops until the token cap loops again when retried unchanged, and the variants change the sampling.
4. **Re-run the notebook.** Finished rows are skipped, so only the failed rows are processed.

| Symptom (the `error` of the record) | What to change |
|---|---|
| `Incomplete response: length` (the output hit `max_tokens`, often a loop) | add a `recovery` variant as above, or raise `max_tokens` (keep it below `max_model_len`) |
| Parse error on a finished answer | re-run the stage (stages 1 and 2 are sampled), or add a `recovery` variant with another `temperature` |
| Out of memory, or the engine does not start | lower `gpu_memory_utilization` or `chunk_size` in `runtime`, or `max_model_len` for the stage |

**Do not edit a stage's own settings** (`temperature`, `max_tokens`, `chat_template_kwargs`, the prompt) after rows are finished. Those settings are part of each record's signature, so every finished row of that stage would be run again. Put the change in `recovery` so that only the failed rows are affected.

## Models and licences

| Model | Licence |
|---|---|
| Qwen3-VL-235B-A22B-Instruct | Apache 2.0 |
| Qwen3.8-Flash-Next | qwen-community-1.0 |
| Qwen3.8-27B | see the model card |

## Layout

```
notebooks/   00-07 as above
src/misc_pipeline/   pipeline code (prompts are read byte-exact from resources/prompts)
resources/   prompts, expected hashes, rules; corpus and reference data are private
configs/models.json  model ids, context lengths, sampling, thinking settings
scripts/run_pipeline.py   command line runner
```
