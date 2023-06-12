
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from dataset.pnr import loader
import models.pnr.losses as losses
import optimizers.pnr.optimizer as optim
from models.pnr.build import build_model

from pytorch_lightning.core import LightningModule
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from utils.lta import distributed as du

class VideoTask(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # Backwards compatibility.
        if isinstance(cfg.MODEL.NUM_CLASSES, int):
            cfg.MODEL.NUM_CLASSES = [cfg.MODEL.NUM_CLASSES]

        if not hasattr(cfg.TEST, "NO_ACT"):
            cfg.TEST.NO_ACT = False

        self.cfg = cfg
        self.save_hyperparameters()
        self.model = build_model(cfg)
        self.loss_fun = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(
            reduction=cfg.MODEL.LOSS_REDUCTION
        )

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step_end(self, training_step_outputs):
        if self.cfg.SOLVER.ACCELERATOR == 'dp':
            training_step_outputs['loss'] = training_step_outputs['loss'].mean()
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
        # if self.model.cfg_recognition:
        #     if self.model.cfg_recognition.SOLVER.ACCELERATOR != "dp":
        #         du.init_distributed_groups(self.model.cfg_recognition)
        try:
            aux_cfg = self.model.cfg_recognition
        except AttributeError:
            aux_cfg = None

        if self.cfg.TRAIN.TRAIN_ENABLE:
            self.train_loader = loader.construct_loader(self.cfg, "train", aux_cfg)
            self.val_loader = loader.construct_loader(self.cfg, "val", aux_cfg)
        if self.cfg.TEST.ENABLE:
            self.test_loader = loader.construct_loader(self.cfg, "val", aux_cfg)

    def configure_optimizers(self):
        optimizer = optim.construct_optimizer(self.model, self.cfg)
        steps_in_epoch = len(self.train_loader)
        if self.cfg.SOLVER.LR_POLICY == "cosine":
            slow_fast_scheduler = CosineAnnealingLR(
                optimizer, self.cfg.SOLVER.MAX_EPOCH * steps_in_epoch, last_epoch=-1
            )
        elif self.cfg.SOLVER.LR_POLICY == "constant":
            slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1)
        else:

            def lr_lambda(step):
                return optim.get_epoch_lr(step / steps_in_epoch, self.cfg)

            slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        scheduler = {"scheduler": slow_fast_scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
