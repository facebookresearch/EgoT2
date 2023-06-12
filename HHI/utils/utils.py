#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import sys
from collections import OrderedDict
from torchtext.vocab import vocab

def build_vocab():
    tokens = ['ttm', 'lam', 'asd', '0', '1']
    v = vocab(OrderedDict([(token, 1) for token in tokens]), specials=["</s>", "<unk>"])
    v.set_default_index(v["<unk>"])
    print('vocab size', len(v))
    print('vocab mapping', v.get_stoi())
    return v


def load_ckpt(backbone, ckpt_name, load_asd=False):
    print(f'Loading pre-trained model: {ckpt_name}')
    ckpt = torch.load(
        ckpt_name,
        map_location=lambda storage, loc: storage,
    )

    def remove_first_module(key):
        return ".".join(key.split(".")[1:])

    key = "state_dict" if "state_dict" in ckpt.keys() else "model_state"
    if load_asd:
        state_dict = {
            remove_first_module(k): v
            for k, v in ckpt.items()
        }
    else:
        state_dict = {
            remove_first_module(k): v
            for k, v in ckpt[key].items()
        }

    missing_keys, unexpected_keys = backbone.load_state_dict(
        state_dict, strict=False
    )

    print('missing', missing_keys)
    print('unexpected', unexpected_keys)


def load_parameters(selfState, ckpt_name):
    print(f'Loading pre-trained model: {ckpt_name}')
    loadedState = torch.load(
        ckpt_name,
        map_location=lambda storage, loc: storage,
    )
    for name, param in loadedState.items():
        origName = name
        if name not in selfState:
            name = name.replace("module.", "")
            if name not in selfState:
                print("%s is not in the model." % origName)
                continue
        if selfState[name].size() != loadedState[origName].size():
            sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s" % (
            origName, selfState[name].size(), loadedState[origName].size()))
            continue
        selfState[name].copy_(param)


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False