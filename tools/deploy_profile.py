#!/usr/bin/env python3
"""Profile one RaCFormer validation sample as a PyTorch deployment baseline."""

import argparse
import contextlib
import importlib
import os
import sys
import time

import mmcv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from mmcv import Config
from mmcv.parallel import DataContainer, MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


class Tee:
    """Write profiling output to both the terminal and a text file."""

    def __init__(self, stream, output_file):
        self.stream = stream
        self.output_file = output_file

    def write(self, text):
        self.stream.write(text)
        self.output_file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.output_file.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile one RaCFormer sample with PyTorch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--out",
        default="outputs/deploy_baseline/profile_result_gpu3.txt")
    args = parser.parse_args()

    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    return args


def describe_value(value, name, indent=0):
    """Recursively print shapes and types without dumping tensor contents."""
    prefix = "  " * indent
    if isinstance(value, DataContainer):
        print(
            "{}{}: DataContainer(stack={}, cpu_only={}, pad_dims={})".format(
                prefix, name, value.stack, value.cpu_only, value.pad_dims))
        describe_value(value.data, "data", indent + 1)
    elif torch.is_tensor(value):
        print(
            "{}{}: Tensor(shape={}, dtype={}, device={})".format(
                prefix, name, tuple(value.shape), value.dtype, value.device))
    elif isinstance(value, np.ndarray):
        print(
            "{}{}: ndarray(shape={}, dtype={})".format(
                prefix, name, value.shape, value.dtype))
    elif isinstance(value, dict):
        print("{}{}: dict(keys={})".format(prefix, name, list(value.keys())))
        for key, item in value.items():
            describe_value(item, str(key), indent + 1)
    elif isinstance(value, (list, tuple)):
        print("{}{}: {}(len={})".format(
            prefix, name, type(value).__name__, len(value)))
        for index, item in enumerate(value):
            describe_value(item, "[{}]".format(index), indent + 1)
    elif hasattr(value, "tensor") and torch.is_tensor(value.tensor):
        print(
            "{}{}: {}(tensor_shape={}, dtype={}, device={})".format(
                prefix, name, type(value).__name__, tuple(value.tensor.shape),
                value.tensor.dtype, value.tensor.device))
    else:
        rendered = repr(value)
        if len(rendered) > 160:
            rendered = rendered[:157] + "..."
        print("{}{}: {} = {}".format(
            prefix, name, type(value).__name__, rendered))


def prepare_batch(dataset, sample_index):
    sample = dataset[sample_index]
    batch = collate([sample], samples_per_gpu=1)
    return sample, batch


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def print_summary(label, summary):
    print(
        "{}: mean={mean:.3f} ms, p50={p50:.3f} ms, "
        "p95={p95:.3f} ms, min={min:.3f} ms, max={max:.3f} ms".format(
            label, **summary))


def extract_prediction(result):
    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError(
            "Expected one prediction in a list, got {}".format(type(result)))
    prediction = result[0]
    if "pts_bbox" in prediction:
        prediction = prediction["pts_bbox"]
    required = ("boxes_3d", "scores_3d", "labels_3d")
    missing = [key for key in required if key not in prediction]
    if missing:
        raise KeyError("Prediction is missing keys: {}".format(missing))
    return prediction


def profile(args):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for profiling")

    torch.cuda.set_device(0)
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True
    cfg = Config.fromfile(args.config)

    # Match val.py: importing these modules registers custom datasets/models.
    importlib.import_module("models")
    importlib.import_module("loaders")

    dataset_cfg = cfg.data[args.split]
    dataset = build_dataset(dataset_cfg)
    if args.sample_index >= len(dataset):
        raise IndexError(
            "sample index {} is outside dataset of length {}".format(
                args.sample_index, len(dataset)))

    model = build_model(cfg.model)
    model.cuda()
    model.eval()
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    checkpoint = load_checkpoint(model, args.weights, map_location="cuda", strict=True)

    # Keep checkpoint-version behavior aligned with val.py when available.
    if "version" in checkpoint:
        from models.utils import VERSION
        VERSION.name = checkpoint["version"]

    print("=== RaCFormer single-sample PyTorch profile ===")
    print("config: {}".format(os.path.abspath(args.config)))
    print("weights: {}".format(os.path.abspath(args.weights)))
    print("split: {}".format(args.split))
    print("dataset length: {}".format(len(dataset)))
    print("sample index: {}".format(args.sample_index))
    print("warmup iterations: {}".format(args.warmup))
    print("profile iterations: {}".format(args.iters))
    print("CUDA device: {}".format(torch.cuda.get_device_name(0)))
    print("PyTorch: {}".format(torch.__version__))
    print("CUDA runtime: {}".format(torch.version.cuda))

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    hook_state = {"capture": False, "timing": False, "kwargs": None}

    def forward_pre_hook(_module, _args, kwargs):
        if hook_state["capture"]:
            hook_state["kwargs"] = kwargs
        if hook_state["timing"]:
            start_event.record()

    def forward_hook(_module, _args, _kwargs, _output):
        if hook_state["timing"]:
            end_event.record()

    pre_hook_handle = model.module.register_forward_pre_hook(
        forward_pre_hook, with_kwargs=True)
    forward_hook_handle = model.module.register_forward_hook(
        forward_hook, with_kwargs=True)

    sample, batch = prepare_batch(dataset, args.sample_index)

    print("\n=== Dataset pipeline sample ===")
    describe_value(sample, "sample")
    print("\n=== Collated batch ===")
    describe_value(batch, "batch")

    # Run through MMDataParallel so MMCV unwraps nested DataContainers exactly
    # as it does in val.py. The pre-hook sees kwargs after that scatter step.
    hook_state["capture"] = True
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **batch)
    torch.cuda.synchronize()
    hook_state["capture"] = False
    print("\n=== Actual model kwargs after CUDA scatter ===")
    describe_value(hook_state["kwargs"], "kwargs")

    print("\n=== Warmup ===")
    with torch.no_grad():
        for _ in range(args.warmup):
            model(return_loss=False, rescale=True, **batch)
    torch.cuda.synchronize()

    hook_state["kwargs"] = None
    del sample, batch, result
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    forward_times = []
    end_to_end_times = []
    result = None

    print("Completed {} warmup iterations".format(args.warmup))
    print("\n=== Timed iterations ===")
    with torch.no_grad():
        for _ in range(args.iters):
            end_to_end_start = time.perf_counter()
            _, current_batch = prepare_batch(dataset, args.sample_index)

            hook_state["timing"] = True
            result = model(return_loss=False, rescale=True, **current_batch)
            hook_state["timing"] = False
            torch.cuda.synchronize()

            end_to_end_times.append(
                (time.perf_counter() - end_to_end_start) * 1000.0)
            forward_times.append(start_event.elapsed_time(end_event))
            del current_batch

    pre_hook_handle.remove()
    forward_hook_handle.remove()

    allocated_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    reserved_mb = torch.cuda.max_memory_reserved() / (1024.0 ** 2)

    print_summary("GPU forward latency", percentile_summary(forward_times))
    print_summary("End-to-end latency", percentile_summary(end_to_end_times))
    print("max_memory_allocated: {:.2f} MB".format(allocated_mb))
    print("max_memory_reserved: {:.2f} MB".format(reserved_mb))

    prediction = extract_prediction(result)
    boxes = prediction["boxes_3d"]
    scores = prediction["scores_3d"]
    labels = prediction["labels_3d"]
    print("\n=== Output structure ===")
    print("result type: {}".format(type(result).__name__))
    print("result length: {}".format(len(result)))
    print("result[0] keys: {}".format(list(result[0].keys())))
    print("boxes_3d.tensor.shape: {}".format(tuple(boxes.tensor.shape)))
    print("scores_3d.shape: {}".format(tuple(scores.shape)))
    print("labels_3d.shape: {}".format(tuple(labels.shape)))
    print("detection count N: {}".format(boxes.tensor.shape[0]))


def main():
    args = parse_args()
    output_path = os.path.abspath(args.out)
    mmcv.mkdir_or_exist(os.path.dirname(output_path))
    with open(output_path, "w", buffering=1) as output_file:
        stdout_tee = Tee(sys.stdout, output_file)
        stderr_tee = Tee(sys.stderr, output_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print("Profiling report: {}".format(output_path))
            profile(args)
            print("\nProfiling completed successfully")


if __name__ == "__main__":
    main()
