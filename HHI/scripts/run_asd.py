#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from configs.asd.config import argparser
from tasks.asd.video_task import ActiveSpeakerDetection
from tasks.asd.video_task_taskspecific import ActiveSpeakerDetection2Loader
from utils.utils import load_ckpt, load_parameters


def main():
    task = ActiveSpeakerDetection2Loader(args) if args.two_loader else ActiveSpeakerDetection(args)
    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, mode="max", save_last=True,
        save_top_k=3, auto_insert_metric_name=True
    )
    trainer = Trainer(
        gpus=1,
        accelerator="gpu",
        callbacks=checkpoint_callback,
        max_epochs=25,
        default_root_dir=os.path.join("./logs/asd", args.output_dir),
        fast_dev_run=args.fast_dev_run,
    )
    if args.eval:
        if args.two_loader:
            load_ckpt(task.model, args.ckpt)
            load_ckpt(task.lossAV, args.ckpt)
        else:
            load_parameters(task.state_dict(), args.ckpt)
        trainer.validate(task)
    elif args.resume:
        trainer.fit(task, ckpt_path=args.ckpt)
    else:
        trainer.fit(task)


if __name__ == '__main__':
    args = argparser.parse_args()
    main()