#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import json
import tqdm
import argparse

from utils.pnr.parser import load_config_file
from dataset.pnr import loader
from models.pnr.build import build_model
from utils.multitask.load_model import load_ckpt
result = []


def get_results(preds, info):
    for pred, uid in zip(preds, info["unique_id"]):
        pred_idx = torch.argmax(pred).item()
        sc = False if pred_idx == 0 else True
        result.append({"unique_id": uid, "state_change": sc})


def evaluate(cfg_name):
    device = torch.device("cuda:0")
    cfg = load_config_file(cfg_name)
    model = build_model(cfg).to(device)
    load_ckpt(model, cfg.MISC.CHECKPOINT_FILE_PATH)
    model.eval()
    cfg_aux = model.cfg_recognition

    val_loader = loader.construct_loader(cfg, "val", cfg_aux)
    correct, total = 0, 0
    for batch in tqdm.tqdm(val_loader):
        frames, labels, state_change_label, fps, info = batch['orig']
        frames_lta = batch['recognition']
        x1 = [frames[0].to(device)]
        x2 = [frames_lta[0].to(device), frames_lta[1].to(device)]
        with torch.no_grad():
            state_change_preds = model(x1, x2)
        state_change_preds = state_change_preds[:, :, 0].cpu()
        tmp = torch.argmax(state_change_preds, dim=1)==state_change_label
        correct += tmp.sum()
        total += len(tmp)
        # print(correct, total, correct.item()/total)

    acc = correct.item() / total
    print('Accuracy', acc)


def submit(cfg_name):
    device = torch.device("cuda:0")
    cfg = load_config_file(cfg_name)
    model = build_model(cfg).to(device)
    load_ckpt(model, cfg.MISC.CHECKPOINT_FILE_PATH)
    model.eval()

    cfg_aux = model.cfg_recognition
    test_loader = loader.construct_loader(cfg, "test", cfg_aux)
    for batch in tqdm.tqdm(test_loader):
        frames, labels, state_change_label, fps, info = batch['orig']
        frames_aux = batch['recognition']
        x1 = [frames[0].to(device)]
        x2 = [frames_aux[0].to(device), frames_aux[1].to(device)]
        with torch.no_grad():
            preds = model(x1, x2)
        get_results(preds, info)
        # break
    json_object = json.dumps(result)
    with open('submit_oscc.json', "w") as f_out:
        f_out.write(json_object)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--eval', action='store_true', help='eval')
    parser.add_argument('--submit', action='store_true', help='submit')
    args = parser.parse_args()

    if args.eval:
        evaluate(args.cfg)

    if args.submit:
        submit(args.cfg)


