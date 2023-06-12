#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import math

from models.lta import losses
from optimizers.lta import lr_scheduler
from utils.lta import distributed as du
from utils.lta import logging
from dataset.lta import loader
from models.lta import build_model
from models.lta.lta_models_lta_transfer import TaskFusionMFTransformer2TaskSeqDecoder
from models.lta.lta_models_seqdecoder import ForecastingEncoderSeqDecoder, ForecastingEncoderSeparateSeqDecoder
from pytorch_lightning.core import LightningModule
from utils.multitask.build_vocab import build_vocab
logger = logging.get_logger(__name__)


class VideoTask(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # Backwards compatibility.
        if isinstance(cfg.MODEL.NUM_CLASSES, int):
            cfg.MODEL.NUM_CLASSES = [cfg.MODEL.NUM_CLASSES]

        if not hasattr(cfg.TEST, "NO_ACT"):
            logger.info("Default NO_ACT")
            cfg.TEST.NO_ACT = False

        if not hasattr(cfg.MODEL, "TRANSFORMER_FROM_PRETRAIN"):
            cfg.MODEL.TRANSFORMER_FROM_PRETRAIN = False

        if not hasattr(cfg.MODEL, "STRIDE_TYPE"):
            cfg.EPIC_KITCHEN.STRIDE_TYPE = "constant"

        self.cfg = cfg
        self.save_hyperparameters()
        if self.cfg.MODEL.MODEL_NAME == "ForecastingEncoderSeqDecoder":
            self.vocab = build_vocab()
            self.model = ForecastingEncoderSeqDecoder(cfg, self.vocab)
        elif self.cfg.MODEL.MODEL_NAME == "ForecastingEncoderSeparateSeqDecoder":
            self.vocab = build_vocab()
            self.model = ForecastingEncoderSeparateSeqDecoder(cfg, self.vocab)
        elif self.cfg.MODEL.MODEL_NAME == "TaskFusionMFTransformer2TaskSeqDecoder":
            self.vocab = build_vocab()
            self.model = TaskFusionMFTransformer2TaskSeqDecoder(cfg, self.vocab)
        else:
            self.model = build_model(cfg)
        self.loss_fun = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(reduction="mean")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step_end(self, training_step_outputs):
        if self.cfg.SOLVER.ACCELERATOR == "dp":
            training_step_outputs["loss"] = training_step_outputs["loss"].mean()
        return training_step_outputs

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def forward(self, inputs):
        return self.model(inputs)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def setup(self, stage):
        # Setup is called immediately after the distributed processes have been
        # registered. We can now setup the distributed process groups for each machine
        # and create the distributed data loaders.
        # if not self.cfg.FBLEARNER:
        if self.cfg.SOLVER.ACCELERATOR != "dp":
            du.init_distributed_groups(self.cfg)
        try:
            aux_cfg = self.model.cfg_pnr
        except AttributeError:
            aux_cfg = None

        if aux_cfg is None:
            try:
                aux_cfg = self.vocab
            except AttributeError:
                aux_cfg = None

        self.train_loader = loader.construct_loader(self.cfg, "train", aux_cfg)
        self.val_loader = loader.construct_loader(self.cfg, "val", aux_cfg)
        # self.test_loader = loader.construct_loader(self.cfg, "test", aux_cfg) ###!!!

    def configure_optimizers(self):
        steps_in_epoch = len(self.train_loader)
        return lr_scheduler.lr_factory(
            self.model, self.cfg, steps_in_epoch, self.cfg.SOLVER.LR_POLICY
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def on_after_backward(self):
        if (
            self.cfg.LOG_GRADIENT_PERIOD >= 0
            and self.trainer.global_step % self.cfg.LOG_GRADIENT_PERIOD == 0
        ):
            for name, weight in self.model.named_parameters():
                if weight is not None:
                    self.logger.experiment.add_histogram(
                        name, weight, self.trainer.global_step
                    )
                    if weight.grad is not None:
                        self.logger.experiment.add_histogram(
                            f"{name}.grad", weight.grad, self.trainer.global_step
                        )
