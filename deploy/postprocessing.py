"""Convert framework-specific detections into deployment output arrays."""

import numpy as np

from .input_schema import DetectionResult


def parse_detection_result(result):
    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError('expected one prediction result')
    prediction = result[0].get('pts_bbox', result[0])
    required = ('boxes_3d', 'scores_3d', 'labels_3d')
    missing = [key for key in required if key not in prediction]
    if missing:
        raise KeyError('prediction is missing {}'.format(missing))

    return DetectionResult(
        boxes_3d=np.asarray(
            prediction['boxes_3d'].tensor.detach().cpu(), dtype=np.float32),
        scores_3d=np.asarray(
            prediction['scores_3d'].detach().cpu(), dtype=np.float32),
        labels_3d=np.asarray(
            prediction['labels_3d'].detach().cpu(), dtype=np.int64))
