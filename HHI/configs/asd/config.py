#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

argparser = argparse.ArgumentParser(description='Ego4D Active Speaker Detection')

argparser.add_argument('--data_file_path', type=str, default='../data/asd', help='data file path')
argparser.add_argument('--two_loader', action='store_true', help='whether to use 2 dataloaders (TTM and LAM same dataloader, ASD another dataloader')
argparser.add_argument('--batch_size', type=int, default=1800, help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
argparser.add_argument('--num_workers', type=int, default=10)

argparser.add_argument('--lr', type=float, default=0.0001)
argparser.add_argument('--lr_decay', type=float, default=0.95)
argparser.add_argument('--init_from_ava', action='store_true')
argparser.add_argument('--finetune', action='store_true')
argparser.add_argument('--nodecay', action='store_true')
argparser.add_argument('--resume', action='store_true')
argparser.add_argument('--fast_dev_run', action='store_true')
argparser.add_argument('--output_dir', type=str, default="debug")
argparser.add_argument('--ckpt', type=str, default='', help='eval ckpt')
argparser.add_argument('--eval', action='store_true', help='eval mode')

argparser.add_argument('--model', required=True, type=str)
argparser.add_argument('--checkpoint', type=str, default='../data/asd/pretrain_AVA.model', help='model pretrained on AVA')
argparser.add_argument('--lam_checkpoint', type=str, default='../pretrained_models/ts_lam.pth', help='LAM checkpoint to load')
argparser.add_argument('--ttm_checkpoint', type=str, default='../pretrained_models/ts_ttm.pth', help='TTM checkpoint to load')
argparser.add_argument('--asd_checkpoint', type=str, default='../pretrained_models/ts_asd.pth', help='ASD checkpoint to load')
argparser.add_argument('--dropout', type=float, default=0.1, help='transformer dropout rate')
argparser.add_argument('--num_layers', type=int, default=1, help='transformer layer num')
argparser.add_argument('--num_heads', type=int, default=4, help='transformer number of heads')
argparser.add_argument('--hidden_dim', type=int, default=256, help='transformer layer hidden dim')
argparser.add_argument('--hidden_dim2', type=int, default=512, help='transformer layer hidden dim')