"""Checkpoint-compatible PyTorch runner without MMDataParallel/DataContainer."""

import copy
import importlib

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

from .input_schema import PreparedBatch
from .postprocessing import parse_detection_result


class RaCFormerPyTorchRunner:
    """Load the training model unchanged and execute deployment batches."""

    def __init__(self, config, weights, device='cuda:0'):
        if not torch.cuda.is_available():
            raise RuntimeError('RaCFormer deployment requires a CUDA device')
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.cfg = Config.fromfile(config)

        importlib.import_module('models')
        importlib.import_module('loaders')

        self.model = build_model(self.cfg.model)
        self.model.to(self.device)
        self.model.eval()
        checkpoint = load_checkpoint(
            self.model, weights, map_location=self.device, strict=True)
        if 'version' in checkpoint:
            from models.utils import VERSION
            VERSION.name = checkpoint['version']

    def prepare(self, batch, non_blocking=False):
        if not isinstance(batch, PreparedBatch):
            raise TypeError('batch must be a PreparedBatch')
        return batch.to(self.device, non_blocking=non_blocking)

    @staticmethod
    def _model_kwargs(batch):
        # The outer lists represent test-time augmentations. Radar points are
        # ordered as augmentation -> temporal frame -> batch item.
        return dict(
            return_loss=False,
            rescale=True,
            img=[batch.image],
            img_metas=[[copy.deepcopy(batch.img_meta)]],
            radar_points=[[[points] for points in batch.radar_points]],
            radar_depth=[batch.radar_depth],
            radar_rcs=[batch.radar_rcs])

    def infer_raw(self, batch):
        with torch.no_grad():
            return self.model(**self._model_kwargs(batch))

    def infer(self, batch):
        return parse_detection_result(self.infer_raw(batch))
