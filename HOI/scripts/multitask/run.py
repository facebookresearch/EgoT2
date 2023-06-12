#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from tasks.multitask.video_task import *
from tasks.multitask.video_task_separate import PnrOnlyTaskTranslation, OsccOnlyTaskTranslation, ActionOnlyTaskTranslation, Task12Translation
from tasks.multitask.video_task_action import Unified4TaskTranslationAction
from configs.multitask.config import argparser
from utils.multitask.build_vocab import build_vocab
from utils.multitask.load_model import load_ckpt

def main():
    vocab = build_vocab()
    if args.task == "unify3task":
        task = Unified3TaskTranslation(args, vocab)
    elif args.task == "unify4task":
        task = Unified4TaskTranslation(args, vocab)
    elif args.task == "unify4taskaction":
        task = Unified4TaskTranslationAction(args, vocab)
    elif args.task == "unify6task":
        task = Unified6TaskTranslationNoPredict(args, vocab) if args.nopredict else Unified6TaskTranslation(args, vocab)
    elif args.task == "pnr_only":
        task = PnrOnlyTaskTranslation(args, vocab)
    elif args.task == "oscc_only":
        task = OsccOnlyTaskTranslation(args, vocab)
    elif args.task == "action_only":
        task = ActionOnlyTaskTranslation(args, vocab)
    elif args.task == "task12":
        task = Task12Translation(args)

    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, mode="min", save_last=True,
        save_top_k=args.save_top_k, auto_insert_metric_name=True
    )
    trainer = Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        strategy="ddp",
        callbacks=checkpoint_callback,
        log_every_n_steps=args.log_every_n_step,
        max_epochs=20,
        default_root_dir=os.path.join("./logs/multitask", args.output_dir),
        fast_dev_run=args.fast_dev_run
    )

    if args.eval:
        load_ckpt(task.model, args.ckpt)
        trainer.validate(task)
    else:
        trainer.fit(task)


if __name__ == '__main__':
    args = argparser.parse_args()
    print(args)
    main()
    print(args)
