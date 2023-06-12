#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


import utils.pnr.logging as logging
from utils.pnr.parser import parse_args, load_config
from tasks.pnr.keyframe_detection import StateChangeAndKeyframeLocalisation, KeyframeLocalisation, KeyframeLocalisationCnnLSTM, StateChangeClassification
from tasks.pnr.video_taskspecific_pnr import StateChangeClassification2Loader, KeyframeLocalisation2Loader
from utils.multitask.load_model import load_ckpt
import warnings
warnings.filterwarnings("ignore")

logger = logging.get_logger(__name__)


def main(cfg):
    if cfg.DATA.TASK == "state_change_detection_and_keyframe_localization":
        TaskType = StateChangeAndKeyframeLocalisation
    elif cfg.DATA.TASK == "keyframe_localization":
        TaskType = KeyframeLocalisation
    elif cfg.DATA.TASK == "keyframe_localization_2loader":
        TaskType = KeyframeLocalisation2Loader
    elif cfg.DATA.TASK == "state_change_classification":
        TaskType = StateChangeClassification
    elif cfg.DATA.TASK == "state_change_classification_2loader":
        TaskType = StateChangeClassification2Loader
    else:
        raise NotImplementedError('Task {} not implemented.'.format(
            cfg.DATA.TASK
        ))

    task = TaskType(cfg)

    trainer = Trainer(
        gpus=cfg.MISC.NUM_GPUS,
        num_nodes=cfg.MISC.NUM_SHARDS,
        accelerator=cfg.SOLVER.ACCELERATOR,
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps=0,
        benchmark=True,
        replace_sampler_ddp=False,
        callbacks=[ModelCheckpoint(
            monitor=task.checkpoint_metric,
            mode="max",
            save_last=True,
            save_top_k=3,
        )],
        fast_dev_run=cfg.MISC.FAST_DEV_RUN,
        default_root_dir=os.path.join(cfg.MISC.OUTPUT_DIR, cfg.MISC.LOG_DIR),
        resume_from_checkpoint=cfg.MISC.CHECKPOINT_FILE_PATH
    )

    if cfg.TRAIN.TRAIN_ENABLE and cfg.TEST.ENABLE:
        trainer.fit(task)
        return trainer.test()

    elif cfg.TRAIN.TRAIN_ENABLE:
        return trainer.fit(task)

    elif cfg.TEST.ENABLE:
        load_ckpt(task.model, cfg.MISC.CHECKPOINT_FILE_PATH)
        return trainer.test(task)


if __name__ == "__main__":
    args = parse_args()
    main(load_config(args))
