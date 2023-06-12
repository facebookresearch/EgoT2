#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import json
import torch
import tqdm
import argparse

from utils.lta.parser import load_config_from_file as load_lta_config
from dataset.lta.long_term_anticipation_lta_auxtask import Ego4dLongTermAnticipationwithAuxTask, Ego4dLongTermAnticipationwithAuxTaskSeparateSequenceLabel
from dataset.lta import loader
from models.lta import build_model
from utils.multitask.load_model import load_ckpt
from utils.multitask.build_vocab import build_vocab
from configs.multitask.config import argparser
from models.multitask.video_model_builder import TaskTranslationPromptTransformer6Task


def evaluate(cfg_name):
    device = torch.device("cuda:0")
    cfg = load_lta_config(cfg_name)
    cfg.NUM_GPUS = 1
    cfg.TRAIN.BATCH_SIZE = 8
    cfg.DATA_LOADER.NUM_WORKERS = 8

    model = build_model(cfg).to(device)
    load_ckpt(model, cfg.CHECKPOINT_FILE_PATH)
    model.eval()

    try:
        aux_cfg = model.cfg_pnr
    except AttributeError:
        aux_cfg = None
    val_loader = loader.construct_loader(cfg, "val", aux_cfg)

    v_correct, n_correct, cnt = 0.0, 0.0, 0.0

    test_cnt = 0
    for batch in tqdm.tqdm(val_loader):
        test_cnt = test_cnt + 1
        inputs, forecast_labels, _, _, label_clip_times, _ = batch['orig']
        inputs2 = batch['pnr']
        x1 = [inputs[0].to(device), inputs[1].to(device)]
        x2 = inputs2.to(device)
        with torch.no_grad():
            preds = model(x1, x2)
        preds_verb = torch.argmax(preds[0][:, 0, :], dim=-1).cpu()
        preds_noun = torch.argmax(preds[1][:, 0, :], dim=-1).cpu()
        labels_verb = forecast_labels[:, 0, 0]
        labels_noun = forecast_labels[:, 0, 1]

        v_correct += (preds_verb == labels_verb).sum()
        n_correct += (preds_noun == labels_noun).sum()
        cnt += preds_verb.shape[0]

    print(f'Verb accuracy {(v_correct / cnt).item()} | Noun accuracy {(n_correct / cnt).item()} | Cnt {cnt}')


def submit(cfg_name):
    cfg = load_lta_config(cfg_name)
    cfg.NUM_GPUS = 1
    model = build_model(cfg).cuda()
    load_ckpt(model, cfg.CHECKPOINT_FILE_PATH)
    dataset = Ego4dLongTermAnticipationwithAuxTask(cfg, model.cfg_pnr, "test")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        sampler=None,
        num_workers=32,
        drop_last=False
    )
    pred_dict = {}
    for batch in tqdm.tqdm(loader):
        inputs, _, _, last_clip_ids, _, _ = batch['orig']
        inputs2 = batch['pnr']
        k = cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT
        x = [inputs[0].cuda(), inputs[1].cuda()]
        x2 = inputs2.cuda()
        preds = model.generate(x, x2, k=k)
        for idx in range(len(last_clip_ids)):
            pred_dict[last_clip_ids[idx]] = {
                'verb': preds[0][idx].cpu().tolist(),
                'noun': preds[1][idx].cpu().tolist(),
            }
    json.dump(pred_dict, open('submit_lta.json', 'w'))


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