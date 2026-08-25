#!/usr/bin/env python3
"""Minimal torchrun/NCCL all-reduce preflight for training launchers."""

import datetime
import os

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl', timeout=datetime.timedelta(seconds=90))
    try:
        value = torch.tensor(
            [float(rank + 1)], device='cuda', dtype=torch.float32)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        expected = world_size * (world_size + 1) / 2.0
        if value.item() != expected:
            raise RuntimeError(
                'rank {} all-reduce expected {}, received {}'.format(
                    rank, expected, value.item()))
        print(
            'rank={} local_rank={} world_size={} device={} sum={} '
            'P2P_DISABLE={} IB_DISABLE={}'.format(
                rank, local_rank, world_size,
                torch.cuda.get_device_name(local_rank), value.item(),
                os.environ.get('NCCL_P2P_DISABLE', ''),
                os.environ.get('NCCL_IB_DISABLE', '')),
            flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    if rank == 0:
        print('NCCL collective preflight: PASS', flush=True)


if __name__ == '__main__':
    main()
