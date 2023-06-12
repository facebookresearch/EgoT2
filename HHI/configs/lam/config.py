#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

argparser = argparse.ArgumentParser(description='Ego4D Looking at me')

# Dataset config
argparser.add_argument('--source_path', type=str, default='../data/lam/video_imgs', help='Video image directory')
argparser.add_argument('--json_path', type=str, default='../data/lam/json_original', help='Face tracklets directory')
argparser.add_argument('--test_path', type=str, default='../data/lam/ocial_test/videos_challenge', help='Test set')
argparser.add_argument('--gt_path', type=str, default='../data/lam/json_original/result_LAM', help='Groundtruth directory')
argparser.add_argument('--train_file', type=str, default='../data/lam/split/train.list', help='Train list')
argparser.add_argument('--val_file', type=str, default='../data/lam/split/val.list', help='Validation list')
argparser.add_argument('--train_stride', type=int, default=13, help='Train subsampling rate')
argparser.add_argument('--val_stride', type=int, default=13, help='Validation subsampling rate')
argparser.add_argument('--test_stride', type=int, default=1, help='Test subsampling rate')

# Training config
argparser.add_argument('--rank', type=int, default=0, help='Raank id')
argparser.add_argument('--epochs', type=int, default=40, help='Maximum epoch')
argparser.add_argument('--batch_size', type=int, default=64, help='Batch size')
argparser.add_argument('--num_gpus', type=int, default=1, help='gpus')
argparser.add_argument('--num_workers', type=int, default=10, help='Num workers')
argparser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
argparser.add_argument('--weights', type=list, default=[0.136, 0.864], help='Class weight')
argparser.add_argument('--fast_dev_run', action='store_true', help='fast dev run')
argparser.add_argument('--exp_path', type=str, default='debug', help='Path to results')
argparser.add_argument('--output_dir', type=str, default='debug', help='Log dir name')
argparser.add_argument('--ckpt', type=str, default='', help='eval ckpt')
argparser.add_argument('--eval', action='store_true', help='eval mode')

# Model config
argparser.add_argument('--model', type=str, required=True, help='Model architecture')
argparser.add_argument('--checkpoint', type=str, help='Checkpoint to load (model initialization)')
