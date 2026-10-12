"""Shared helpers (CPU only): configuration, file layout, result validation, metrics, printed values."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C_MIN, C_MAX = 4.0, 12.0  # bundled fields are stored in [-1, 1]; the experiments use [C_MIN, C_MAX] model units
START = 8.0  # constant starting model
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
CIRCLE_RANGE = (4.000406265258789, 11.998579025268555)  # normalization of the circle training set


def config():
    return json.loads((ROOT / 'configs/experiments.json').read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_path(prior):
    """Bundled prior checkpoint, or the file named by PNPFLOW_CKPT_FLAT / PNPFLOW_CKPT_GAUSS."""
    record = json.loads((ROOT / 'configs/checkpoints.json').read_text())[prior]
    return Path(os.environ.get('PNPFLOW_CKPT_' + prior.upper(), ROOT / 'checkpoints' / record['filename']))


def target(cohort, image):
    """Target field of one test image on the 64 x 63 solver grid (last column dropped), in model units."""
    with np.load(ROOT / config()['cohorts'][cohort]['data']) as z:
        return (0.5 * (z['c'][image] + 1.0) * (C_MAX - C_MIN) + C_MIN)[:, :-1]


def settings(cohort, method, image, seed=None):
    """Settings of one reconstruction, as recorded in its result file."""
    cfg = config()
    cohort_cfg = cfg['cohorts'][cohort]
    s = dict(cohort=cohort, method=method, image=image, start=START)
    if method == 'pnp':
        s.update({k: v for k, v in cfg['pnp'].items() if k != 'seeds'}, seed=seed, prior=cohort_cfg['prior'])
    else:
        s.update(updates=cfg['tv']['updates'], weight=cohort_cfg['tv_weight'], step=cohort_cfg['tv_step'])
    return s


def reconstruction_path(root, cohort, method, image, seed=None):
    name = f'image{image:02d}_seed{seed}.npz' if method == 'pnp' else f'image{image:02d}.npz'
    return Path(root) / cohort / method / name


def validate(z, cohort, method, image, seed=None):
    """Check a result file: finite 64 x 63 fields, the bundled target, and the configured settings."""
    for key in ('c_true', 'c_est'):
        if z[key].shape != (64, 63) or not np.isfinite(z[key]).all():
            raise ValueError(f'Invalid {key} field')
    if not np.array_equal(z['c_true'], target(cohort, image)):
        raise ValueError('Reconstruction target differs from the bundled input')
    if str(z['settings']) != json.dumps(settings(cohort, method, image, seed), sort_keys=True):
        raise ValueError('Recorded settings differ from configs/experiments.json')


def rel_l2_pct(a, truth):
    """Whole-image relative L2 error in percent."""
    return 100 * float(np.linalg.norm(a - truth) / np.linalg.norm(truth))


def lesion_contrast(field, tmask, cmask):
    """Mean over the eroded target core minus mean over the 2-5 pixel ring that excludes all inclusions."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    core = binary_erosion(tmask, structure=CROSS)
    if not core.any():
        core = tmask
    dist = distance_transform_edt(~tmask)
    ring = (dist >= 2) & (dist <= 5) & ~cmask
    return float(field[core].mean() - field[ring].mean())


def breast_masks(path):
    """Target-lesion and all-lesion masks of the breast patches on the solver grid."""
    with np.load(path) as z:
        return z['target_mask'][:, :, :-1].astype(bool), z['circle_mask'][:, :, :-1].astype(bool)


def circle_inclusions():
    """Inclusion masks of the 50 circle targets (shared by both circle families) on the solver grid.

    Inclusion pixels differ from the background value 8 in the constant-background target recovered with
    the training-set normalization; each 4-connected component is one inclusion. Returns the all-inclusion
    masks (50 x 64 x 63) and, per target, the list of component masks.
    """
    from scipy import ndimage
    with np.load(ROOT / config()['cohorts']['flat']['data']) as z:
        lo, hi = CIRCLE_RANGE
        c = ((z['c'].astype(np.float64) + 1) / 2 * (hi - lo) + lo).astype(np.float32)
    mask = ~np.isclose(c, START, atol=2e-6, rtol=0)
    records = json.loads((ROOT / 'data/gauss/records_test.json').read_text())
    assert [int(m.sum()) for m in mask] == [r['circle_pixels'] for r in records], 'mask rule disagrees with the records'
    mask = mask[:, :, :-1]
    parts = []
    for m in mask:
        labels, n = ndimage.label(m)
        parts.append([labels == k for k in range(1, n + 1)])
    return mask, parts


def inclusion_contrast_error(field, truth, parts, allmask):
    """Mean over a target's inclusions of |contrast(field) - contrast(truth)|."""
    return float(np.mean([abs(lesion_contrast(field, p, allmask) - lesion_contrast(truth, p, allmask)) for p in parts]))


def normalize_printed(s):
    """Map a printed value to a comparable ASCII token ('+0.03' -> '0.03', '-0.00' -> '0.00')."""
    s = str(s).strip().replace('−', '-').replace(',', '')
    if s.startswith('+'):
        s = s[1:]
    return '0' + s[2:] if s.startswith('-0') and set(s[2:]) <= set('.0') else s


def as_printed(value, expected):
    """Format a computed value with the precision of the printed one."""
    exp = normalize_printed(expected)
    if isinstance(value, str):
        return normalize_printed(value)
    decimals = len(exp.split('.')[1]) if '.' in exp else 0
    return normalize_printed(f'{float(value):.{decimals}f}')
