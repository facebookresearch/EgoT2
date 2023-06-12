#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch


def load_ckpt(backbone, ckpt_name):
    print(f'Loading pre-trained model: {ckpt_name}')
    ckpt = torch.load(
        ckpt_name,
        map_location=lambda storage, loc: storage,
    )

    def remove_first_module(key):
        return ".".join(key.split(".")[1:])

    key = "state_dict" if "state_dict" in ckpt.keys() else "model_state"

    state_dict = {
        remove_first_module(k): v
        for k, v in ckpt[key].items()
    }

    missing_keys, unexpected_keys = backbone.load_state_dict(
        state_dict, strict=False
    )

    print('missing', missing_keys)
    print('unexpected', unexpected_keys)


def load_checkpoint(model, ckpt):
    print(f'Loading pre-trained model: {ckpt}')
    try:
        model.load_state_dict(torch.load(ckpt)['state_dict'])
    except RuntimeError:
        print('Loading the model by modifying the keys...')
        # When the model is trained using data parallel class
        state_dict = torch.load(ckpt)['state_dict']
        new_state_dict = dict()
        for key, value in state_dict.items():
            new_key = key.replace('model.', '').replace('pnr_', 'pnr_model.').replace('oscc_', 'oscc_model.').replace('recognition_', 'recognition_model.')
            new_state_dict[new_key] = value  # module.
        model.load_state_dict(new_state_dict)


def load_recognition_backbone(backbone, ckpt_name):
    print(f'Loading pre-trained model: {ckpt_name}')
    ckpt = torch.load(
        ckpt_name,
        map_location=lambda storage, loc: storage,
    )

    def remove_first_module(key):
        return ".".join(key.split(".")[1:])

    key = "state_dict" if "state_dict" in ckpt.keys() else "model_state"

    state_dict = {
        remove_first_module(k): v
        for k, v in ckpt[key].items()
        if "head" not in k
    }

    missing_keys, unexpected_keys = backbone.load_state_dict(
        state_dict, strict=False
    )

    print('missing', missing_keys)
    print('unexpected', unexpected_keys)

    # Ensure only head key is missing.w
    # assert len(unexpected_keys) == 0
    # assert all(["head" in x for x in missing_keys])

    # for key in missing_keys:
    #     print(f"Could not load {key} weights")


def load_lta_backbone(model, ckpt_name, replace_backbone=False, skip_head=False):
    print(f'Loading pre-trained model: {ckpt_name}')
    ckpt = torch.load(
        ckpt_name,
        map_location=lambda storage, loc: storage,
    )
    new_state_dict = dict()
    for k, v in ckpt["state_dict"].items():
        if skip_head:
            if "head" in k:
                continue
        new_k = k.replace("model.", "")
        if replace_backbone:
            new_k = new_k.replace("backbone.", "")
        new_state_dict[new_k] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print('missing', missing_keys)
    print('unexpected', unexpected_keys)



def freeze_backbone_params(model):
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
        else:
            print(name, 'requires grad')


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

