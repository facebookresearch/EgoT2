#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from configs.ttm.config import argparser
from tasks.ttm.video_task import TalkingToMe
from tasks.ttm.video_task_2loader import TalkingToMe2Loader


def main():
    task = TalkingToMe2Loader(args) if args.two_loader else TalkingToMe(args)
    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, mode="max", save_last=True,
        save_top_k=3, auto_insert_metric_name=True
    )
    trainer = Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        callbacks=checkpoint_callback,
        max_epochs=args.epochs,
        default_root_dir=os.path.join("./logs/ttm", args.output_dir),
        fast_dev_run=args.fast_dev_run,
    )
    if args.eval:
        task = task.load_from_checkpoint(checkpoint_path=args.ckpt, args=args)
        trainer.validate(task)
    elif args.submit:
        task = task.load_from_checkpoint(checkpoint_path=args.ckpt, args=args)
        trainer.test(task)
    else:
        trainer.fit(task)


if __name__ == '__main__':
    args = argparser.parse_args()
    args.exp_path = os.path.join("./logs/ttm", args.output_dir)
    main()