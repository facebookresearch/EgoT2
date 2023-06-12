#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import argparse

argparser = argparse.ArgumentParser(description='Ego4D Multi-task translation')
argparser.add_argument('--num_gpus', type=int, default=8)

argparser.add_argument('--ratio1', type=float, default=1.0)
argparser.add_argument('--ratio2', type=float, default=1.0)
argparser.add_argument('--ratio3', type=float, default=1.0)
argparser.add_argument('--ratio4', type=float, default=1.0)
argparser.add_argument('--ratio5', type=float, default=1.0)
argparser.add_argument('--ratio6', type=float, default=1.0)
argparser.add_argument('--output_dir', type=str, default="debug")
argparser.add_argument('--ckpt', type=str, default="")
argparser.add_argument('--ckpt_dir', type=str, default="")
argparser.add_argument('--task', type=str, default="unify6task")
argparser.add_argument('--model', type=str, default="default")
argparser.add_argument('--eval_task', type=str, default="all")
argparser.add_argument('--nopredict', action='store_true')
argparser.add_argument('--optim', type=str, default="default")
argparser.add_argument('--lr', type=float, default=1e-4)
argparser.add_argument('--wd', type=float, default=1e-4)
argparser.add_argument('--check_val_every', type=int, default=1)
argparser.add_argument('--limit_train_batches', type=float, default=1.0)
argparser.add_argument('--log_every_n_step', type=int, default=50)
argparser.add_argument('--save_top_k', type=int, default=3)
argparser.add_argument('--eval', action='store_true')
argparser.add_argument('--fast_dev_run', action='store_true')

argparser.add_argument('--pnr_cfg_file', type=str, default="./configs/eval/pnr.yaml")
argparser.add_argument('--oscc_cfg_file', type=str, default="./configs/eval/oscc.yaml")
argparser.add_argument('--action_cfg_file', type=str, default="./configs/eval/recognition.yaml")
argparser.add_argument('--lta_cfg_file', type=str, default="./configs/eval/lta.yaml")

argparser.add_argument('--pnr_ft', action='store_true')
argparser.add_argument('--oscc_ft', action='store_true')
argparser.add_argument('--action_ft', action='store_true')
argparser.add_argument('--oscc_no_temp_pool', action='store_true')

argparser.add_argument('--batch_size', type=int, default=4)
argparser.add_argument('--num_workers', type=int, default=4)
argparser.add_argument('--loader_mode', type=str, default="max_size_cycle")   #min_size

argparser.add_argument('--hidden_dim', type=int, default=512)  # 256
argparser.add_argument('--ff_dim', type=int, default=1024)
argparser.add_argument('--num_heads', type=int, default=8)
argparser.add_argument('--num_layers', type=int, default=3)
argparser.add_argument('--dropout', type=float, default=0.1)
argparser.add_argument('--ts_token', type=int, default=0)

