#!/usr/bin/env python3
"""
Loss function
"""

#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch.nn as nn

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "mse": nn.MSELoss
}


def get_loss_func(loss_name):
    """
    Retrieve the loss fucntion given the loss name

    Args (str):
        loss_name: name of the loss to use
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
