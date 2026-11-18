#!/usr/bin/env python
"""Command-line version of notebooks 01-06.

    python scripts/run_pipeline.py all            # pass1 -> pass2 -> pass3 -> stage1 -> stage2 -> submission
    python scripts/run_pipeline.py stage2 submission
    python scripts/run_pipeline.py jitter          # optional second submission (not part of 'all')
    python scripts/run_pipeline.py pass1 --limit 20      # rehearsal on the first 20 rows (use MISC_WORK=work_rehearsal)

Each stage loads its own model, writes to work/<stage>/ and frees the GPUs when it ends.
Environment variables: MISC_DATA (data folder), MISC_WORK (output folder), MISC_BACKEND=mock (placeholder outputs, no GPU).
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from misc_pipeline import common, runner  # noqa: E402

ORDER = ['pass1', 'pass2', 'pass3', 'stage1', 'stage2', 'submission']


def run(stage, limit):
    D = common.STAGE_DIRS
    if stage == 'pass1':
        from misc_pipeline import pass1
        jobs = pass1.build_jobs(limit)
        print(runner.run_stage('pass1', jobs, D['pass1'], pass1.parse, label_images=False)); print(pass1.export(jobs))
    elif stage == 'pass2':
        from misc_pipeline import pass2
        jobs = pass2.build_jobs(limit)
        print(runner.run_stage('pass2', jobs, D['pass2'], pass2.parse)); print(pass2.export(jobs))
    elif stage == 'pass3':
        from misc_pipeline import pass3
        jobs = pass3.build_jobs(limit)
        print(runner.run_stage('pass3', jobs, D['pass3'], pass3.parse)); print(pass3.export(jobs))
    elif stage == 'stage1':
        from misc_pipeline import stage1, stage_exports
        jobs, metas = stage1.build_jobs(limit)
        print(runner.run_stage('stage1', jobs, D['stage1'], stage1.parse_record)); print(stage_exports.export_stage1(jobs, metas))
    elif stage == 'stage2':
        from misc_pipeline import stage2, stage_exports
        jobs, metas = stage2.build_jobs(limit)
        print(runner.run_stage('stage2', jobs, D['stage2'], stage2.parse_record))
        print(stage_exports.export_stage2(jobs, metas, limit or len(common.test_rows())))
    elif stage == 'submission':
        from misc_pipeline import submission
        submission.make()
    elif stage == 'jitter':
        from misc_pipeline import jitter
        jitter.make()
    else:
        raise SystemExit('unknown stage ' + stage)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stages', nargs='+', help='all | ' + ' | '.join(ORDER))
    ap.add_argument('--limit', type=int, default=None, help='first N test rows only')
    a = ap.parse_args()
    for s in (ORDER if a.stages == ['all'] else a.stages):
        print('=' * 20, s, '=' * 20, flush=True)
        run(s, a.limit)
