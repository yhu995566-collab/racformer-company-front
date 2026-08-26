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
