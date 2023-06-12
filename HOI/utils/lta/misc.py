#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os

import numpy as np
import psutil
import torch
from fvcore.nn.flop_count import flop_count
from fvcore.nn.precise_bn import update_bn_stats

from .import logging
from .batchnorm_helper import SubBatchNorm3d
from .datasets_utils import pack_pathway_output

logger = logging.get_logger(__name__)


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def get_flop_stats(model, cfg, is_train):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.

    Returns:
        float: the total number of gflops of the given model.
    """
    rgb_dimension = 3
    if is_train:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_FRAMES,
            cfg.DATA.TRAIN_CROP_SIZE,
            cfg.DATA.TRAIN_CROP_SIZE,
        )
    else:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_FRAMES,
            cfg.DATA.TEST_CROP_SIZE,
            cfg.DATA.TEST_CROP_SIZE,
        )
    flop_inputs = pack_pathway_output(cfg, input_tensors)
    for i in range(len(flop_inputs)):
        flop_inputs[i] = flop_inputs[i].unsqueeze(0).cuda(non_blocking=True)

    # If detection is enabled, count flops for one proposal.
    if cfg.DATA.TASK == "detection":
        bbox = torch.tensor([[0, 0, 1.0, 0, 1.0]])
        bbox = bbox.cuda()
        inputs = (flop_inputs, bbox)
    else:
        inputs = (flop_inputs,)

    gflop_dict, _ = flop_count(model, inputs)
    gflops = sum(gflop_dict.values())
    return gflops


def log_model_info(model, cfg, is_train=True):
    """
    Log info, includes number of parameters, gpu usage and gflops.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        is_train (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    logger.info("FLOPs: {:,} GFLOPs".format(get_flop_stats(model, cfg, is_train)))
    logger.info("nvidia-smi")
    os.system("nvidia-smi")


def aggregate_split_bn_stats(module):
    """
    Recursively find all SubBN modules and aggregate sub-BN stats.
    Args:
        module (nn.Module)
    Returns:
        count (int): number of SubBN module found.
    """
    count = 0
    for child in module.children():
        if isinstance(child, SubBatchNorm3d):
            child.aggregate_stats()
            count += 1
        else:
            count += aggregate_split_bn_stats(child)
    return count


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def calculate_and_update_precise_bn_2loader(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for batch in loader:
            inputs, *_ = batch['orig']
            inputs2 = batch['pnr']
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(inputs2, (list,)):
                for i in range(len(inputs2)):
                    inputs2[i] = inputs2[i].cuda(non_blocking=True)
            else:
                inputs2 = inputs2.cuda(non_blocking=True)
            yield inputs, inputs2

    # Update the bn stats.
    update_bn_stats_2loader(model, _gen_loader(), num_iters)

from torch import nn
from fvcore.nn.precise_bn import get_bn_modules, _PopulationVarianceEstimator
import itertools
import tqdm
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type


@torch.no_grad()
def update_bn_stats_2loader(
    model: nn.Module,
    data_loader: Iterable[Any],
    num_iters: int = 200,
    progress: Optional[str] = None,
) -> None:
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.
    See Sec. 3 of the paper "Rethinking Batch in BatchNorm" for details.

    Args:
        model (nn.Module): the model whose bn stats will be recomputed.

            Note that:

            1. This function will not alter the training mode of the given model.
               Users are responsible for setting the layers that needs
               precise-BN to training mode, prior to calling this function.

            2. Be careful if your models contain other stateful layers in
               addition to BN, i.e. layers whose state can change in forward
               iterations.  This function will alter their state. If you wish
               them unchanged, you need to either pass in a submodule without
               those layers, or backup the states.
        data_loader (iterator): an iterator. Produce data as inputs to the model.
        num_iters (int): number of iterations to compute the stats.
        progress: None or "tqdm". If set, use tqdm to report the progress.
    """
    bn_layers = get_bn_modules(model)

    if len(bn_layers) == 0:
        return
    logger.info(f"Computing precise BN statistics for {len(bn_layers)} BN layers ...")

    # In order to make the running stats only reflect the current batch, the
    # momentum is disabled.
    # bn.running_mean = (1 - momentum) * bn.running_mean + momentum * batch_mean
    # Setting the momentum to 1.0 to compute the stats without momentum.
    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    batch_size_per_bn_layer: Dict[nn.Module, int] = {}

    def get_bn_batch_size_hook(
        module: nn.Module, input: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        assert (
            module not in batch_size_per_bn_layer
        ), "Some BN layers are reused. This is not supported and probably not desired."
        x = input[0]
        assert isinstance(
            x, torch.Tensor
        ), f"BN layer should take tensor as input. Got {input}"
        # consider spatial dimensions as batch as well
        batch_size = x.numel() // x.shape[1]
        batch_size_per_bn_layer[module] = batch_size
        return (x,)

    hooks_to_remove = [
        bn.register_forward_pre_hook(get_bn_batch_size_hook) for bn in bn_layers
    ]

    estimators = [
        _PopulationVarianceEstimator(bn.running_mean, bn.running_var)
        for bn in bn_layers
    ]

    ind = -1
    for inputs, inputs2 in tqdm.tqdm(
        itertools.islice(data_loader, num_iters),
        total=num_iters,
        disable=progress != "tqdm",
    ):
        ind += 1
        batch_size_per_bn_layer.clear()
        model(inputs, inputs2)

        for i, bn in enumerate(bn_layers):
            # Accumulates the bn stats.
            batch_size = batch_size_per_bn_layer.get(bn, None)
            if batch_size is None:
                continue  # the layer was unused in this forward
            estimators[i].update(bn.running_mean, bn.running_var, batch_size)
    assert ind == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, "
        "but the dataloader stops at {} iterations.".format(num_iters, ind)
    )

    for i, bn in enumerate(bn_layers):
        # Sets the precise bn stats.
        bn.running_mean = estimators[i].pop_mean
        bn.running_var = estimators[i].pop_var
        bn.momentum = momentum_actual[i]
    for hook in hooks_to_remove:
        hook.remove()