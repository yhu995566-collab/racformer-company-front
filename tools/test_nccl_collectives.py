#!/usr/bin/env python3
"""Fail-fast single-node NCCL collective test used before long training."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    values = torch.full(
        (4 * 1024 * 1024,), float(rank + 1),
        dtype=torch.float32, device=local_rank)
    if rank == 0:
        values.fill_(7.0)
    dist.broadcast(values, src=0)
    if values[0].item() != 7.0 or values[-1].item() != 7.0:
        raise RuntimeError("NCCL broadcast produced inconsistent values")

    values.fill_(float(rank + 1))
    dist.all_reduce(values)
    expected = world_size * (world_size + 1) / 2
    if values[0].item() != expected or values[-1].item() != expected:
        raise RuntimeError("NCCL all_reduce produced inconsistent values")

    rank_value = torch.tensor([rank], dtype=torch.int64, device=local_rank)
    gathered = [torch.empty_like(rank_value) for _ in range(world_size)]
    dist.all_gather(gathered, rank_value)
    actual = [int(item.item()) for item in gathered]
    if actual != list(range(world_size)):
        raise RuntimeError("NCCL all_gather produced {}".format(actual))

    dist.barrier()
    print("rank {}/{} NCCL collectives OK".format(rank, world_size))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
