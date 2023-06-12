#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from configs.lam.config import argparser
from tasks.lam.video_task import LookingAtMe


def main():
    task = LookingAtMe(args)
    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, mode="max", save_last=True,
        save_top_k=3, auto_insert_metric_name=True
    )
    trainer = Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        callbacks=checkpoint_callback,
        max_epochs=args.epochs,
        default_root_dir=os.path.join("./logs/lam", args.output_dir),
        fast_dev_run=args.fast_dev_run,
    )
    if args.eval:
        task = task.load_from_checkpoint(checkpoint_path=args.ckpt, args=args)
        trainer.validate(task)
    else:
        trainer.fit(task)
        trainer.validate(task)


if __name__ == '__main__':
    args = argparser.parse_args()
    args.exp_path = os.path.join("./logs/lam", args.output_dir)
    main()