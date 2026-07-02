import os
from os import path as osp
import utils
import logging
import argparse
import importlib
import torch
import torch.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed, multi_gpu_test, single_gpu_test
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from models.utils import VERSION


def evaluate(dataset, results, **kwargs):
    metrics = dataset.evaluate(results, **kwargs)
    logging.info('--- Evaluation Results ---')
    for name, value in metrics.items():
        if isinstance(value, (float, int)):
            logging.info('%s: %.4f', name, value)
        else:
            logging.info('%s: %s', name, value)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--out', help='Optional PKL path for raw predictions')
    parser.add_argument('--score-threshold', type=float, default=0.1)
    parser.add_argument('--nms-iou-threshold', type=float, default=0.2)
    parser.add_argument('--bev-iou-threshold', type=float, default=0.5)
    parser.add_argument('--iou-3d-threshold', type=float, default=0.5)
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--override', nargs='+', action=DictAction)
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if local_rank == 0:
        utils.init_logging(None, cfgs.debug)
    else:
        logging.root.disabled = True

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist.init_process_group('nccl', init_method='env://')

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    dataset_cfg = cfgs.data[args.split]
    logging.info('Loading %s set from %s', args.split, dataset_cfg.data_root)
    val_dataset = build_dataset(dataset_cfg)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()

    if world_size > 1:
        model = MMDistributedDataParallel(model, [local_rank], broadcast_buffers=False)
    else:
        model = MMDataParallel(model, [0])

    logging.info('Loading checkpoint from %s' % args.weights)
    checkpoint = load_checkpoint(
        model, args.weights, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR)
    )

    if 'version' in checkpoint:
        VERSION.name = checkpoint['version']

    if world_size > 1:
        results = multi_gpu_test(model, val_loader, gpu_collect=False)
    else:
        results = single_gpu_test(model, val_loader)

    if local_rank == 0:
        if args.out:
            mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.out)))
            mmcv.dump(results, args.out)
            logging.info('Predictions saved to %s', args.out)
        if not args.skip_eval:
            eval_kwargs = {}
            if val_dataset.__class__.__name__ == 'CompanyFrontDataset':
                eval_kwargs = dict(
                    score_threshold=args.score_threshold,
                    nms_iou_threshold=args.nms_iou_threshold,
                    bev_iou_threshold=args.bev_iou_threshold,
                    iou_3d_threshold=args.iou_3d_threshold)
            evaluate(val_dataset, results, **eval_kwargs)


if __name__ == '__main__':
    main()
