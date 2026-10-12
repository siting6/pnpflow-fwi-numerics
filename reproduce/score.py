"""Measure the 660 reconstructions of the tables from a directory of result files (the archive or new runs)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import (ROOT, START, breast_masks, circle_inclusions, config, inclusion_contrast_error,
                    lesion_contrast, reconstruction_path, rel_l2_pct, sha256, target, validate)


def score(root, output, archive=False):
    cfg = config()
    original = json.loads((ROOT / 'results/sources.json').read_text())['sha256']
    allmask, parts = circle_inclusions()
    hashes, rows = {}, []
    for cohort, cohort_cfg in cfg['cohorts'].items():
        if sha256(ROOT / cohort_cfg['data']) != cohort_cfg['data_sha256']:
            raise ValueError(f'Bundled input changed: {cohort}')
        masks = breast_masks(ROOT / cohort_cfg['data']) if cohort == 'breast' else None
        for image in range(cohort_cfg['n_images']):
            truth = target(cohort, image)
            if masks is not None:
                true_contrast = lesion_contrast(truth, masks[0][image], masks[1][image])

            def errors(field):
                contrast = (abs(lesion_contrast(field, masks[0][image], masks[1][image]) - true_contrast)
                            if masks is not None else
                            inclusion_contrast_error(field, truth, parts[image], allmask[image]))
                return dict(l2_pct=rel_l2_pct(field, truth), contrast_error=contrast)

            rows.append(dict(cohort=cohort, image=image, method='start', seed='',
                             **errors(np.full_like(truth, START))))
            for method, seed in [('pnp', s) for s in cfg['pnp']['seeds']] + [('tv', None)]:
                path = reconstruction_path(root, cohort, method, image, seed)
                relative = path.relative_to(root).as_posix()
                hashes[relative] = sha256(path)
                if archive and hashes[relative] != original.get(relative):
                    raise ValueError(f'Archive checksum differs: {relative}')
                with np.load(path) as z:
                    validate(z, cohort, method, image, seed)
                    rows.append(dict(cohort=cohort, image=image, method=method,
                                     seed='' if seed is None else seed, **errors(z['c_est'])))
        print(f'{cohort}: scored {cohort_cfg["n_images"]} targets')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'measurements.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cohort', 'image', 'method', 'seed', 'l2_pct', 'contrast_error'],
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (output / 'sources.json').write_text(json.dumps({'sha256': hashes}, indent=2) + '\n')
    print(f'Wrote {len(rows)} measurement rows from {len(hashes)} reconstructions to {output}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path, help='root of the result files: <cohort>/<method>/image<i>[_seed<s>].npz')
    ap.add_argument('--archive', action='store_true', help='require the checksums recorded in results/sources.json')
    ap.add_argument('--output', type=Path, default=ROOT / 'runs/scored')
    args = ap.parse_args()
    score(args.directory.resolve(), args.output, args.archive)


if __name__ == '__main__':
    main()
