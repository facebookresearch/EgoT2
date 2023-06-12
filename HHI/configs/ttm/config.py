#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
argparser = argparse.ArgumentParser(description='Ego4D Talking to me')

# Dataset config
argparser.add_argument('--test_data_path', type=str, default='../data/ttm/final_test_data')
argparser.add_argument('--seg_info', type=str, default='../data/ttm/seg_info.json')
argparser.add_argument('--img_path', type=str, default='../data/ttm/video_imgs', help='Video image directory')
argparser.add_argument('--wave_path', type=str, default='../data/ttm/wave', help='Audio wave directory')
argparser.add_argument('--gt_path', type=str, default='../data/ttm/result_TTM', help='Groundtruth directory')
argparser.add_argument('--json_path', type=str, default='../data/ttm/json_original', help='Face tracklets directory')
argparser.add_argument('--train_file', type=str, default='../data/ttm/split/train.list', help='Train list')
argparser.add_argument('--val_file', type=str, default='../data/ttm/split/val.list', help='Validation list')
argparser.add_argument('--test_file', type=str, default='../data/ttm/split/test.list', help='Test list')
argparser.add_argument('--train_stride', type=int, default=3, help='Train subsampling rate')
argparser.add_argument('--val_stride', type=int, default=1, help='Validation subsampling rate')
argparser.add_argument('--test_stride', type=int, default=1, help='Test subsampling rate')

# Training config
argparser.add_argument('--two_loader', action='store_true', help='whether to use 2 dataloaders (TTM and LAM same dataloader, ASD another dataloader')
argparser.add_argument('--rank', type=int, default=0, help='Rank id')
argparser.add_argument('--epochs', type=int, default=40, help='Maximum epoch')
argparser.add_argument('--batch_size', type=int, default=400, help='Batch size')
argparser.add_argument('--num_gpus', type=int, default=1, help='transformer layer num')
argparser.add_argument('--num_workers', type=int, default=10, help='Num workers')
argparser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
argparser.add_argument('--wd', type=float, default=0, help='weight decay')
argparser.add_argument('--weights', type=list, default=[0.266, 0.734], help='Class weight')
argparser.add_argument('--fast_dev_run', action='store_true', help='fast dev run')
argparser.add_argument('--exp_path', type=str, default='debug', help='Path to results')
argparser.add_argument('--output_dir', type=str, default='debug', help='Log dir name')
argparser.add_argument('--ckpt', type=str, default='', help='eval ckpt')
argparser.add_argument('--eval', action='store_true', help='eval mode')
argparser.add_argument('--submit', action='store_true', help='submit (test)')

# Model config
argparser.add_argument('--model', type=str, required=True, help='Model architecture')
argparser.add_argument('--checkpoint', type=str, help='TTM Checkpoint to load')
argparser.add_argument('--lam_checkpoint', type=str, default='../pretrained_models/ts_lam.pth', help='LAM checkpoint to load')
argparser.add_argument('--ttm_checkpoint', type=str, default='../pretrained_models/ts_ttm.pth', help='TTM checkpoint to load')
argparser.add_argument('--asd_checkpoint', type=str, default='../pretrained_models/ts_asd.pth', help='ASD checkpoint to load')
argparser.add_argument('--nofreeze', action='store_true', help='Train all or not')
argparser.add_argument('--dropout', type=float, default=0.1, help='transformer dropout rate')
argparser.add_argument('--num_layers', type=int, default=3, help='transformer layer num')
argparser.add_argument('--num_heads', type=int, default=4, help='transformer number of heads')
argparser.add_argument('--hidden_dim', type=int, default=256, help='transformer layer hidden dim')
argparser.add_argument('--hidden_dim2', type=int, default=512, help='transformer layer hidden dim')
