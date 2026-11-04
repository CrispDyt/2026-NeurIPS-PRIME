"""Checks for archival identity and table aggregation, without long training runs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_tables', ROOT/'scripts/build_tables.py')
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


def test_statistics_population_and_missing():
    result = tables.statistics([1., 2., 5., None])
    assert result['n_valid'] == 3
    assert result['n_missing'] == 1
    assert result['mean'] == pytest.approx(8/3)
    assert result['std'] == pytest.approx(np.sqrt(26/9))
    assert [result['Q25'], result['median'], result['Q75']] == [1.5, 2., 3.5]
    assert tables.statistics([5., None])['std'] == 0
    assert tables.statistics([None])['mean'] is None


def test_reference_tables_complete_and_repaired():
    obs, _, seeds = tables.collect(ROOT/'reference')
    pooled, per_seed, appendix = tables.summarize(obs, seeds)
    assert (len(pooled), len(per_seed), len(appendix)) == (108, 540, 108)
    assert sum(r['n_missing'] for r in pooled if r['metric'].startswith('DRTD')) == 11
    ks_mse = {r['metric']: r for r in pooled if r['system'] == 'KS' and r['model'] == 'MSE'}
    # Independent audit anchors distinguish the corrected five-seed result from the old four-seed table.
    assert ks_mse['TCM']['mean'] == pytest.approx(88.02802248604942)
    assert ks_mse['TCM']['std'] == pytest.approx(114.18893111356735)
    assert ks_mse['DRTD_Sigma3']['mean'] == pytest.approx(9.798990088, rel=1e-8)
    single = next(r for r in pooled if (r['system'], r['model'], r['metric']) == ('Lorenz63', 'MSE', 'DRTD_Sigma2'))
    assert single['n_valid'] == 1 and single['std'] == 0


def test_checkpoint_repair_identity():
    path = ROOT/'checkpoints/ks/partial_seed_123123/node_BEST.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '90f996baf2db51e18c6655e5f189ae7bd54065bd53ed59f64b3e1b727cfb7872'
    other = ROOT/'checkpoints/ks/partial_seed_123123123/node_BEST.pt'
    assert path.read_bytes() != other.read_bytes()


def test_asset_manifest():
    path = ROOT/'reference/assets.sha256.json'
    assert path.exists(), 'Missing repository asset inventory'
    records = json.loads(path.read_text())
    assert len(records) >= 60
    actual = {str(p.relative_to(ROOT)) for name in ('data', 'checkpoints')
              for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert {record['path'] for record in records} == actual
    for record in records:
        asset = ROOT/record['path']
        assert asset.stat().st_size == record['bytes'], str(asset)
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == record['sha256'], str(asset)


def test_transport_metric_does_not_silently_change_on_failure(monkeypatch):
    import pandas  # initialize before the legacy pickle aliases in the CFD module
    spec = importlib.util.spec_from_file_location('cfd_evaluate', ROOT/'experiments/cylinder/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = np.array([[-1., 0.], [1., 0.]], dtype=np.float32)
    second = first + np.array([0., 1.], dtype=np.float32)
    assert module.sinkhorn_w2_cloud(first, second) == pytest.approx(.5, abs=1e-6)

    def unavailable(*args, **kwargs):
        raise RuntimeError('transport backend failed')

    monkeypatch.setattr(module, 'SamplesLoss', unavailable)
    with pytest.raises(RuntimeError, match='transport backend failed'):
        module.sinkhorn_w2_cloud(first, second)
