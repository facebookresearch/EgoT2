#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import json
import tqdm
import argparse
import numpy as np

from utils.pnr.parser import load_config_file
from dataset.pnr import loader
from models.pnr.build import build_model
from utils.multitask.load_model import load_ckpt
from evaluation.pnr.metrics import keyframe_distance
result = []


def get_results(preds, info):
    for pred, start_frame, end_frame, uid in zip(preds, info['clip_start_frame'], info['clip_end_frame'], info["unique_id"]):
        pred_idx = torch.argmax(pred).item()
        print(pred_idx)
        pnr_frame = (end_frame - start_frame) / 16 * pred_idx
        result.append({"unique_id": uid, "pnr_frame": pnr_frame.item()})


def evaluate(cfg_name):
    device = torch.device("cuda:0")
    cfg = load_config_file(cfg_name)
    model = build_model(cfg).to(device)
    load_ckpt(model, cfg.MISC.CHECKPOINT_FILE_PATH)
    model.eval()
    cfg_aux = model.cfg_recognition

    val_loader = loader.construct_loader(cfg, "val", cfg_aux)
    result = []
    for batch in tqdm.tqdm(val_loader):
        frames, labels, state_change_label, fps, info = batch['orig']
        frames_lta = batch['recognition']
        x1 = [frames[0].to(device)]
        x2 = [frames_lta[0].to(device), frames_lta[1].to(device)]
        with torch.no_grad():
            keyframe_preds = model(x1, x2)
        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_label,
            fps,
            info
        )
        result.append(keyframe_avg_time_dist)
    print('PNR error', np.mean(result))


def submit(cfg_name):
    device = torch.device("cuda:0")
    cfg = load_config_file(cfg_name)
    model = build_model(cfg).to(device)
    load_ckpt(model, cfg.MISC.CHECKPOINT_FILE_PATH)
    model.eval()
    cfg_aux = model.cfg_recognition

    test_loader = loader.construct_loader(cfg, "test", cfg_aux)

    for batch in tqdm.tqdm(test_loader):
        frames, labels, state, fps, info_new = batch['orig']
        frames_lta = batch['recognition']
        x1 = [frames[0].to(device)]
        x2 = [frames_lta[0].to(device), frames_lta[1].to(device)]
        with torch.no_grad():
            preds = model(x1, x2)
        get_results(preds, info_new)

    json_object = json.dumps(result)
    with open('submit_pnr.json', "w") as f_out:
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



