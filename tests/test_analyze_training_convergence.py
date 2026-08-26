from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_analyzer():
    path = ROOT / 'tools' / 'analyze_training_convergence.py'
    spec = importlib.util.spec_from_file_location('convergence_analyzer', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_and_classify_plateau(tmp_path):
    analyzer = load_analyzer()
    log = tmp_path / 'train.log'
    lines = []
    values = {2: 0.10, 4: 0.20, 6: 0.201, 8: 0.202, 10: 0.201}
    for epoch, metric in values.items():
        lines.append(
            'Epoch [{}/36][90/100] loss: {:.2f}, loss_cls: 0.10, '
            'loss_bbox: 0.20, loss_depth: 0.30\n'.format(
                epoch, 1.0 / epoch))
        lines.append(
            'company/car_3D_AP@0.5: {:.4f}\n'.format(metric))
    log.write_text(''.join(lines))
    losses, metrics, nonfinite = analyzer.parse_log(log)
    eval_points = [
        (epoch, metric[analyzer.DEFAULT_PRIMARY])
        for epoch, metric in sorted(metrics.items())]
    loss_points = [
        (epoch, values[0]['loss'])
        for epoch, values in sorted(losses.items())]
    status, _ = analyzer.classify(
        eval_points, loss_points, patience=3, min_delta=0.005)
    assert status == 'CONVERGED_PLATEAU'
    assert not nonfinite


def test_classify_improving():
    analyzer = load_analyzer()
    status, _ = analyzer.classify(
        [(2, 0.10), (4, 0.12), (6, 0.15), (8, 0.19)],
        [(2, 2.0), (4, 1.8), (6, 1.6), (8, 1.5)],
        patience=3, min_delta=0.005)
    assert status == 'STILL_IMPROVING'


def test_small_intermediate_gains_do_not_move_significance_baseline():
    analyzer = load_analyzer()
    status, detail = analyzer.classify(
        [(2, 0.0053), (4, 0.0027), (6, 0.0069),
         (8, 0.0035), (10, 0.0119), (12, 0.0104)],
        [(2, 260.0), (4, 255.0), (6, 252.0),
         (8, 250.0), (10, 249.0), (12, 248.0)],
        patience=3, min_delta=0.005)
    assert status == 'STILL_IMPROVING'
    assert 'stale_evals=1' in detail


def test_info_prefix_is_not_reported_as_infinite(tmp_path):
    analyzer = load_analyzer()
    log = tmp_path / 'train.log'
    log.write_text(
        '[2026-08-26][INFO] - Epoch [1/36][90/100] loss: 1.0\n'
        '[2026-08-26][INFO] - Epoch [2/36][90/100] loss: inf\n')
    _, _, nonfinite = analyzer.parse_log(log)
    assert nonfinite == [2]
