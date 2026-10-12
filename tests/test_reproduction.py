"""CPU checks of the table aggregation, the scoring conventions, the experiment plan and result validation."""
import csv
import json

import numpy as np
import pytest

from common import (ROOT, circle_inclusions, config, inclusion_contrast_error, lesion_contrast, rel_l2_pct,
                    settings, target, validate)
from make_tables import build, check
from run_experiments import tasks


def rows_from(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_all_main_table_values():
    values, markdown = build(ROOT / 'results/measurements.csv')
    count, failures = check(values)
    assert count == 24 and failures == []
    assert all(f'Table 4.{i}' in markdown for i in (1, 2, 3))


def test_tables_require_all_four_seeds(tmp_path):
    rows = [r for r in rows_from(ROOT / 'results/measurements.csv')
            if not (r['cohort'] == 'flat' and r['image'] == '0' and r['method'] == 'pnp' and r['seed'] == '3')]
    write_rows(tmp_path / 'incomplete.csv', rows)
    with pytest.raises(ValueError, match='measurement rows'):
        build(tmp_path / 'incomplete.csv')


def test_contrast_aggregation_takes_the_median_error_over_seeds(tmp_path):
    rows = rows_from(ROOT / 'results/measurements.csv')
    for row in rows:
        if row['cohort'] == 'breast' and row['method'] == 'pnp':
            row['contrast_error'] = [0.0, 0.1, 0.7, 0.8][int(row['seed'])]
    write_rows(tmp_path / 'contrast.csv', rows)
    values, _ = build(tmp_path / 'contrast.csv')
    assert values['tab:breast-main/pnp2500_run/contrast'] == pytest.approx(0.4)


def test_experiment_plan_covers_every_target_and_seed():
    selected = tasks(['flat', 'gauss', 'breast'])
    assert len(selected) == 660 and len({t.output for t in selected}) == 660
    for cohort, cfg in config()['cohorts'].items():
        pnp = {(t.image, t.seed) for t in selected if t.cohort == cohort and t.method == 'pnp'}
        tv = {t.image for t in selected if t.cohort == cohort and t.method == 'tv'}
        assert pnp == {(i, s) for i in range(cfg['n_images']) for s in range(4)}
        assert tv == set(range(cfg['n_images']))


def test_invalid_subset_fails_before_launching():
    with pytest.raises(ValueError, match='outside'):
        tasks(['breast'], images=[32])
    with pytest.raises(ValueError, match='seeds'):
        tasks(['flat'], seeds=[4])


def test_relative_error_and_lesion_contrast():
    truth = np.full((64, 63), 8.0, np.float32)
    estimate = truth.copy()
    estimate[0, 0] += 8.0
    assert rel_l2_pct(estimate, truth) == pytest.approx(100 / np.sqrt(64 * 63), rel=1e-6)
    yy, xx = np.mgrid[:64, :63]
    lesion = (yy - 32) ** 2 + (xx - 30) ** 2 <= 6.5 ** 2
    other = (yy - 32) ** 2 + (xx - 41) ** 2 <= 2 ** 2
    field = np.where(lesion, 9.0, 8.0).astype(np.float32)
    field[other] = 20.0
    assert lesion_contrast(field, lesion, lesion | other) == pytest.approx(1.0)


def test_circle_geometry_and_exact_reconstruction():
    masks, components = circle_inclusions()
    assert sum(map(len, components)) == 102
    for mask, parts in zip(masks, components):
        assert np.array_equal(np.logical_or.reduce(parts), mask)
    field = np.where(masks[0], 10.0, 8.0).astype(np.float32)
    assert inclusion_contrast_error(field, field, components[0], masks[0]) == 0.0


def test_validation_rejects_other_settings_and_targets():
    truth = target('flat', 3)
    assert truth.shape == (64, 63) and truth.dtype == np.float32
    good = dict(c_true=truth, c_est=truth, settings=np.asarray(json.dumps(settings('flat', 'tv', 3), sort_keys=True)))
    validate(good, 'flat', 'tv', 3)
    other = dict(settings('flat', 'tv', 3), updates=5000)
    with pytest.raises(ValueError, match='settings'):
        validate(dict(good, settings=np.asarray(json.dumps(other, sort_keys=True))), 'flat', 'tv', 3)
    with pytest.raises(ValueError, match='target'):
        validate(good, 'flat', 'tv', 4)
    with pytest.raises(ValueError, match='settings'):
        validate(dict(good, settings=np.asarray(json.dumps(settings('flat', 'pnp', 3, 0), sort_keys=True))),
                 'flat', 'pnp', 3, 1)
