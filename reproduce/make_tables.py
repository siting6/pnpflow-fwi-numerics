"""Build Tables 4.1-4.3 from the per-run errors and check them against the printed values."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import ROOT, as_printed, config


def build(measurements):
    cfg = config()
    seeds = cfg['pnp']['seeds']
    with Path(measurements).open() as f:
        rows = list(csv.DictReader(f))
    expected_rows = sum(c['n_images'] * (len(seeds) + 2) for c in cfg['cohorts'].values())
    if len(rows) != expected_rows:
        raise ValueError(f'Expected {expected_rows} measurement rows, got {len(rows)}')
    if len({(r['cohort'], r['image'], r['method'], r['seed']) for r in rows}) != len(rows):
        raise ValueError('Duplicate measurement rows')
    values, md = {}, ['# Three main tables', '']
    for number, (cohort, cohort_cfg) in enumerate(cfg['cohorts'].items(), 1):
        errors = {}
        for method in ('start', 'pnp', 'tv'):
            a = np.empty((cohort_cfg['n_images'], 2))
            for image in range(cohort_cfg['n_images']):
                rr = [r for r in rows if r['cohort'] == cohort and int(r['image']) == image and r['method'] == method]
                wanted = {str(s) for s in seeds} if method == 'pnp' else {''}
                if {r['seed'] for r in rr} != wanted or len(rr) != len(wanted):
                    raise ValueError(f'Missing or unexpected runs for {cohort} image {image} {method}')
                metrics = np.array([[float(r['l2_pct']), float(r['contrast_error'])] for r in rr])
                if not np.isfinite(metrics).all() or (metrics < 0).any():
                    raise ValueError(f'Invalid errors for {cohort} image {image} {method}')
                a[image] = np.median(metrics, axis=0)  # median over the PnP-Flow seeds of one target
            errors[method] = a
        md += [f'## Table 4.{number}: {cohort_cfg["title"]}', '',
               '| Reconstruction | Median relative L2 (%) | Median contrast error |', '|---|---:|---:|']
        for method, key, title in [('start', 'start', 'Constant start, m=8'),
                                   ('pnp', 'pnp2500_run', 'PnP-Flow'), ('tv', 'tv2500', 'TV-FWI')]:
            l2, contrast = np.median(errors[method], axis=0)  # median over the targets
            values[f'tab:{cohort}-main/{key}/l2'] = float(l2)
            values[f'tab:{cohort}-main/{key}/contrast'] = float(contrast)
            md.append(f'| {title} | {l2:.2f} | {contrast:.3f} |')
        wins = (errors['pnp'] < errors['tv']).sum(axis=0)
        values[f'text:{cohort}/pnp_lower_l2'] = f'{wins[0]}/{cohort_cfg["n_images"]}'
        values[f'text:{cohort}/pnp_lower_contrast'] = f'{wins[1]}/{cohort_cfg["n_images"]}'
        md += ['', f'PnP-Flow has lower error on {wins[0]}/{cohort_cfg["n_images"]} targets for relative L2 '
               f'and {wins[1]}/{cohort_cfg["n_images"]} for contrast.', '']
    return values, '\n'.join(md) + '\n'


def check(values):
    expected = json.loads((ROOT / 'results/expected.json').read_text())
    failures = [f'{key}: computed {values.get(key)!r}, expected {printed}'
                for key, printed in expected.items()
                if key not in values or as_printed(values[key], printed) != printed]
    return len(expected), failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--measurements', type=Path, default=ROOT / 'results/measurements.csv')
    ap.add_argument('--output', type=Path, default=ROOT / 'results/tables.md')
    ap.add_argument('--check', action='store_true', help='compare with results/expected.json')
    args = ap.parse_args()
    values, markdown = build(args.measurements)
    if args.check:
        total, failures = check(values)
        for failure in failures:
            print('MISMATCH', failure)
        print(f'{total - len(failures)}/{total} printed values reproduced (18 table entries and 6 win counts)')
        if failures:
            raise SystemExit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
