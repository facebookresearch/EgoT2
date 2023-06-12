#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

argparser = argparse.ArgumentParser(description='Ego4D HHI Multitask training config')

argparser.add_argument('--lam_source_path', type=str, default='../data/lam/video_imgs', help='Video image directory')
argparser.add_argument('--lam_json_path', type=str, default='../data/lam/json_original', help='Face tracklets directory')
argparser.add_argument('--lam_gt_path', type=str, default='../data/lam/result_LAM', help='Groundtruth directory')
argparser.add_argument('--lam_train_file', type=str, default='../data/lam/split/train.list', help='Train list')
argparser.add_argument('--lam_val_file', type=str, default='../data/lam/split/val.list', help='Validation list')
argparser.add_argument('--lam_train_stride', type=int, default=13, help='Train subsampling rate')
argparser.add_argument('--lam_val_stride', type=int, default=13, help='Validation subsampling rate')

argparser.add_argument('--ttm_img_path', type=str, default='../data/ttm/video_imgs', help='Video image directory')
argparser.add_argument('--ttm_wave_path', type=str, default='../data/ttm/wave', help='Audio wave directory')
argparser.add_argument('--ttm_gt_path', type=str, default='../data/ttm/result_TTM', help='Groundtruth directory')
argparser.add_argument('--ttm_json_path', type=str, default='../data/ttm/json_original', help='Face tracklets directory')
argparser.add_argument('--ttm_train_file', type=str, default='../data/ttm/split/train.list', help='Train list')
argparser.add_argument('--ttm_val_file', type=str, default='../data/ttm/split/val.list', help='Validation list')
argparser.add_argument('--ttm_train_stride', type=int, default=3, help='Train subsampling rate')
argparser.add_argument('--ttm_val_stride', type=int, default=1, help='Validation subsampling rate')

argparser.add_argument('--asd_file_path', type=str, default='../data/asd', help='asd data file path')

argparser.add_argument('--lam_train_batch_size', type=int, default=64, help='LAM batch size')
argparser.add_argument('--lam_val_batch_size', type=int, default=16, help='LAM batch size')
argparser.add_argument('--ttm_batch_size', type=int, default=15, help='TTM batch size')
argparser.add_argument('--asd_batch_size', type=int, default=600, help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
argparser.add_argument('--num_workers', type=int, default=10)

argparser.add_argument('--num_layers', type=int, default=3, help='transformer layer num')
argparser.add_argument('--num_heads', type=int, default=4, help='transformer number of heads')
argparser.add_argument('--hidden_dim', type=int, default=256, help='transformer layer hidden dim')
argparser.add_argument('--dropout', type=float, default=0.1, help='transformer layer hidden dim')
argparser.add_argument('--ratio1', type=float, default=1.0)
argparser.add_argument('--ratio2', type=float, default=1.0)
argparser.add_argument('--ratio3', type=float, default=1.0)
argparser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
argparser.add_argument('--wd', type=float, default=0, help='weight decay')

argparser.add_argument('--epochs', type=int, default=20, help='Maximum epoch')
argparser.add_argument('--rank', type=int, default=0, help='Rank id')
argparser.add_argument('--num_gpus', type=int, default=1, help='num of gpus')
argparser.add_argument('--eval', action='store_true', help='eval')
argparser.add_argument('--task_translation', action='store_true', help='task translation')
argparser.add_argument('--fast_dev_run', action='store_true', help='fast dev run')
argparser.add_argument('--save_top_k', type=int, default=3, help='save top k')
argparser.add_argument('--output_dir', type=str, default='debug', help='Path to results')
argparser.add_argument('--exp_path', type=str, default='debug', help='Path to results')

argparser.add_argument('--lam_checkpoint', type=str, default='../pretrained_models/ts_lam.pth', help='LAM checkpoint to load')
argparser.add_argument('--ttm_checkpoint', type=str, default='../pretrained_models/ts_ttm.pth', help='TTM checkpoint to load')
argparser.add_argument('--asd_checkpoint', type=str, default='../pretrained_models/ts_asd.pth', help='ASD checkpoint to load')