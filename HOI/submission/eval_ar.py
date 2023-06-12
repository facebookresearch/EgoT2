#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import tqdm
import torch
import argparse
from models.lta import build_model
from dataset.lta import loader
from utils.lta.parser import load_config_from_file as load_lta_config
from utils.multitask.load_model import load_ckpt


def evaluate(cfg_name):
    device = torch.device("cuda:0")
    cfg = load_lta_config(cfg_name)
    cfg.NUM_GPUS = 1
    cfg.TRAIN.BATCH_SIZE = 32
    cfg.DATA_LOADER.NUM_WORKERS = 32

    model = build_model(cfg).to(device)
    load_ckpt(model, cfg.CHECKPOINT_FILE_PATH)
    model.eval()

    val_loader = loader.construct_loader(cfg, "val", model.cfg_pnr)
    print(len(val_loader))

    v_correct, n_correct, cnt = 0.0, 0.0, 0.0
    for batch in tqdm.tqdm(val_loader):
        inputs, labels, _, _ = batch['orig']
        inputs2 = batch['pnr']
        x = [inputs[0].to(device), inputs[1].to(device)]
        x2 = [inputs2[0].to(device)]
        with torch.no_grad():
            preds = model(x, x2)
            pred_verb = torch.argmax(preds[0], dim=-1).cpu()
            pred_noun = torch.argmax(preds[1], dim=-1).cpu()
            v_correct += (pred_verb == labels[:, 0]).sum()
            n_correct += (pred_noun == labels[:, 1]).sum()
            cnt += pred_verb.shape[0]
    print(f'Verb accuracy {(v_correct / cnt).item()} | Noun accuracy {(n_correct / cnt).item()} | Cnt {cnt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    evaluate(args.cfg)
