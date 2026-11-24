import numpy as np

from read_eareog.labels import line_classes, return_sweep_labels


def test_label_boundaries():
    on = np.array([10.0, 20.0, 30.0, 40.0, 2.0])
    dx = np.array([-300.0, -300.0001, -500.0, -500.0, -900.0])
    dy = np.array([0.0, 0.0, 150.0, 149.999, 0.0])
    assert list(return_sweep_labels(on, dx, dy, 5.0, 95.0)) == [20.0, 40.0]


def _stub(pitch=30.0):
    fix, sac = [], []
    t = 10.0
    y = 100.0
    for line in range(6):
        for j, x in enumerate((150, 350, 550, 750, 950, 1150)):
            fix.append((t, 0.2, x, y)); t += 0.2
            if j < 5:
                sac.append((t, 0.03, 200.0, 0.0)); t += 0.03
        dy = pitch if line != 3 else 2 * pitch
        sac.append((t, 0.05, -1000.0, dy)); t += 0.05
        y += dy
    for j, x in enumerate((150, 350, 550)):                  # first fixations of the last line
        fix.append((t, 0.2, x, y)); t += 0.23
    F = {k: np.array([f[i] for f in fix]) for i, k in enumerate(('onset', 'duration', 'x', 'y'))}
    S = {k: np.array([s[i] for s in sac]) for i, k in enumerate(('onset', 'duration', 'dx', 'dy'))}
    return S, F, t


def test_line_classes_on_a_stub_page():
    S, F, t_end = _stub()
    cls = list(line_classes(S, F, 9.0, t_end, 30.0)['cls'])
    assert cls.count('NL') == 5 and cls.count('SKIP') == 1 and cls.count('NONLEFT') == 30


def test_same_line_regression_and_page_precedence():
    fix = [(0.2 * i, 0.15, 100.0 + 60 * i, 500.0) for i in range(20)]          # filler: x range 100 ... 1240
    fix += [(10.0, 0.2, 600, 300), (10.25, 0.2, 700, 300), (10.5, 0.2, 800, 300)]
    sac = [(10.7, 0.03, -150.0, 2.0)]                                            # small regression, same line
    fix += [(10.73, 0.2, 650, 302), (10.98, 0.2, 750, 302), (11.23, 0.2, 850, 302)]
    sac.append((20.0, 0.05, 400.0, -500.0))                                      # rightward but page-level (dy < -400)
    fix += [(20.05, 0.2, 1200, -198)]
    fix.sort()
    F = {k: np.array([f[i] for f in fix]) for i, k in enumerate(('onset', 'duration', 'x', 'y'))}
    S = {k: np.array([s[i] for s in sac]) for i, k in enumerate(('onset', 'duration', 'dx', 'dy'))}
    assert list(line_classes(S, F, 0.0, 30.0, 30.0)['cls']) == ['SAME', 'PAGE']
