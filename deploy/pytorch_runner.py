"""Checkpoint-compatible PyTorch runner without MMDataParallel/DataContainer."""

import copy
import importlib

import torch
import torch.backends.cudnn as cudnn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.models import build_model

from .input_schema import PreparedBatch
from .postprocessing import parse_detection_result


class _PerForwardRadarTemporalCache:
    """Reuse query-independent radar temporal features within one forward."""

    def __init__(self, model):
        self.encoder = model.pts_bbox_head.transformer.decoder.decoder_layer \
            .sampling_radar_bev.temporal_encoder
        self.original_forward = self.encoder.forward
        self.active = False
        self.cached_key = None
        self.cached_output = None
        self.encoder.forward = self.forward

    def begin(self):
        self.active = True
        self.cached_key = None
        self.cached_output = None

    def end(self):
        self.active = False
        self.cached_key = None
        self.cached_output = None

    def forward(self, bev_feats):
        if not self.active:
            return self.original_forward(bev_feats)
        key = (bev_feats.data_ptr(), bev_feats._version, tuple(bev_feats.shape))
        if key != self.cached_key:
            self.cached_output = self.original_forward(bev_feats)
            self.cached_key = key
        return self.cached_output


class RaCFormerPyTorchRunner:
    """Load the training model unchanged and execute deployment batches."""

    def __init__(self, config, weights, device='cuda:0',
                 cache_radar_temporal=False):
        if not torch.cuda.is_available():
            raise RuntimeError('RaCFormer deployment requires a CUDA device')
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        set_random_seed(0, deterministic=True)
        cudnn.benchmark = True
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
        self.radar_temporal_cache = None
        if cache_radar_temporal:
            self.radar_temporal_cache = _PerForwardRadarTemporalCache(
                self.model)

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
        if self.radar_temporal_cache is not None:
            self.radar_temporal_cache.begin()
        try:
            with torch.no_grad():
                return self.model(**self._model_kwargs(batch))
        finally:
            if self.radar_temporal_cache is not None:
                self.radar_temporal_cache.end()

    def infer(self, batch):
        return parse_detection_result(self.infer_raw(batch))
