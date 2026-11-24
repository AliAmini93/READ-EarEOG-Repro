import json

from read_eareog.__main__ import main
from read_eareog.recording import save_recording
from read_eareog.synthetic import synthetic_cohort


def test_run_and_causal_on_saved_synthetic_recordings(tmp_path, capsys):
    for rec in synthetic_cohort(3, minutes=3.0):                     # SYNTHETIC test fixtures; causal needs > 60 s
        save_recording(rec, tmp_path / 'recordings' / rec.name)
    dirs = sorted(str(d) for d in (tmp_path / 'recordings').iterdir())
    assert main(['run', '--data', *dirs, '--out', str(tmp_path / 'r.json')]) == 0
    again = json.loads((tmp_path / 'r.json').read_text())
    assert again['all_synthetic'] and 'SYNTHETIC' in again['notice'] and again['package_version']
    assert again['input_order'] == [d.split('/')[-1] for d in dirs]
    assert main(['run', '--causal', '--data', *dirs, '--out', str(tmp_path / 'c.json')]) == 0
    assert 'f1_symmetric' in json.loads((tmp_path / 'c.json').read_text())['cohort']


def test_errors_are_reported_not_raised(tmp_path, capsys):
    bad = tmp_path / 'p.toml'
    bad.write_text('[detector]\ntheta_min = "0.5"\n')
    assert main(['check-config', '--config', str(bad)]) == 2
    assert 'must be a number' in capsys.readouterr().err
