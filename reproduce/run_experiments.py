"""Run the 528 PnP-Flow and 132 TV-FWI reconstructions of Tables 4.1-4.3, one GPU process per run."""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from common import ROOT, checkpoint_path, config, reconstruction_path, sha256, validate


@dataclass
class Task:
    cohort: str
    method: str
    image: int
    seed: object
    output: Path


def tasks(cohorts, method='both', images=None, seeds=None, out=ROOT / 'runs'):
    cfg = config()
    all_seeds = cfg['pnp']['seeds']
    seeds = all_seeds if seeds is None else seeds
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(all_seeds):
        raise ValueError(f'PnP seeds must be distinct values from {all_seeds}')
    if images is not None and len(set(images)) != len(images):
        raise ValueError('Image indices must be distinct')
    selected = []
    for cohort in cohorts:
        n = cfg['cohorts'][cohort]['n_images']
        for image in range(n) if images is None else images:
            if not 0 <= image < n:
                raise ValueError(f'{cohort}: image {image} is outside 0..{n - 1}')
            for arm in ('pnp', 'tv') if method == 'both' else (method,):
                for seed in seeds if arm == 'pnp' else [None]:
                    selected.append(Task(cohort, arm, image, seed, reconstruction_path(out, cohort, arm, image, seed)))
    return selected


def preflight(selected):
    cfg = config()
    for cohort in {t.cohort for t in selected}:
        if sha256(ROOT / cfg['cohorts'][cohort]['data']) != cfg['cohorts'][cohort]['data_sha256']:
            raise ValueError(f'Input checksum differs: {cohort}')
    records = json.loads((ROOT / 'configs/checkpoints.json').read_text())
    for prior in {cfg['cohorts'][t.cohort]['prior'] for t in selected if t.method == 'pnp'}:
        path = checkpoint_path(prior)
        if not path.is_file():
            raise FileNotFoundError(f'Missing {prior} prior checkpoint; see configs/checkpoints.json')
        if path.stat().st_size != records[prior]['size_bytes'] or sha256(path) != records[prior]['sha256']:
            raise ValueError(f'Checkpoint checksum differs: {prior}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', nargs='+', choices=list(config()['cohorts']), default=list(config()['cohorts']))
    ap.add_argument('--method', choices=['pnp', 'tv', 'both'], default='both')
    ap.add_argument('--images', nargs='+', type=int, help='zero-based target indices (default: all)')
    ap.add_argument('--seeds', nargs='+', type=int, help='PnP-Flow seeds (default: 0 1 2 3)')
    ap.add_argument('--out', type=Path, default=ROOT / 'runs')
    ap.add_argument('--dry-run', action='store_true', help='list the plan without running')
    args = ap.parse_args()
    selected = tasks(args.cohort, args.method, args.images, args.seeds, args.out.resolve())
    for (cohort, method), count in Counter((t.cohort, t.method) for t in selected).items():
        print(f'{cohort} {method}: {count} reconstructions')
    print(f'Total: {len(selected)} reconstructions', flush=True)
    if args.dry_run:
        return
    preflight(selected)
    env = {**os.environ, 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}
    for i, task in enumerate(selected, 1):
        label = (f'[{i}/{len(selected)}] {task.cohort} {task.method} image {task.image}'
                 + (f' seed {task.seed}' if task.method == 'pnp' else ''))
        if task.output.exists():
            with np.load(task.output) as z:
                validate(z, task.cohort, task.method, task.image, task.seed)
            print(label, '(already complete)', flush=True)
            continue
        print(label, flush=True)
        command = [sys.executable, '-u', str(ROOT / 'reproduce/reconstruct.py'), '--cohort', task.cohort,
                   '--method', task.method, '--image', str(task.image), '--output', str(task.output)]
        if task.method == 'pnp':
            command += ['--seed', str(task.seed)]
        task.output.parent.mkdir(parents=True, exist_ok=True)
        with task.output.with_suffix('.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Reconstruction failed; see {task.output.with_suffix(".log")}')
        with np.load(task.output) as z:
            validate(z, task.cohort, task.method, task.image, task.seed)


if __name__ == '__main__':
    main()
