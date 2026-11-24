import numpy as np

from read_eareog.pipeline import attribute_detections, run_primary


def test_end_to_end_on_synthetic(cohort):
    res = run_primary(cohort)
    assert res['all_synthetic'] and res['cohort']['n_recordings'] == len(cohort)
    for r in res['recordings']:
        assert r['f1'] > r['f1_random']
        assert 0.5 <= r['theta'] <= 6.0
    again = run_primary(cohort)
    assert [r['f1'] for r in again['recordings']] == [r['f1'] for r in res['recordings']]
    assert [r['f1_random'] for r in again['recordings']] == [r['f1_random'] for r in res['recordings']]


def test_attribution_counts_every_detection(cohort):
    res = run_primary(cohort)
    r = res['recordings'][0]
    cnt = attribute_detections(r['detections'], cohort[0], 29.5)
    assert sum(cnt.values()) == r['n_detections']
    assert cnt['NL'] + cnt['SKIP'] > 0


def test_duplicate_names_and_exclusions(cohort):
    import pytest
    from dataclasses import replace
    with pytest.raises(ValueError, match='unique'):
        run_primary([cohort[0], cohort[0]])
    few = replace(cohort[1], name='few', window=(cohort[1].window[0], cohort[1].window[0] + 12.0))
    res = run_primary([cohort[0], cohort[2], few])
    assert [e['recording'] for e in res['excluded']] == ['few'] and 'labels' in res['excluded'][0]['reason']
    assert res['input_order'] == [cohort[0].name, cohort[2].name, 'few'] and res['protocol']['refractory_s'] == 0.6


def test_causal_variant_runs(cohort):
    from read_eareog.pipeline import run_causal
    res = run_causal(cohort)
    for r in res['recordings']:
        assert r['f1_asymmetric'] > r['f1_random_asymmetric']
        assert 0 < r['median_latency_ms'] < 500
