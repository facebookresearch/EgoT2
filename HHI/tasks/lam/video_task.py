#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
from pytorch_lightning.core import LightningModule
from torch.utils.data import DistributedSampler
from dataset.lam.data_loader import NewImagerLoader
from models.lam.build import build_model
from utils.lam.utils import get_transform, PostProcessor


class LookingAtMe(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.checkpoint_metric = "val_mAP"
        self.model = build_model(args)
        class_weights = torch.FloatTensor(args.weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def on_train_epoch_start(self):
        if self.args.num_gpus > 1:
            self.train_dataloader.sampler.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        source_frame, target = batch
        output = self.model(source_frame)
        loss = self.criterion(output, target.squeeze(1))
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self.postprocess = PostProcessor(self.args)

    def validation_step(self, batch, batch_idx):
        source_frame, target = batch
        output = self.model(source_frame)
        # loss = self.criterion(output, target.squeeze(1))
        # self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.postprocess.update(output, target)

    def validation_epoch_end(self, outputs):
        self.postprocess.save()
        mAP, acc = self.postprocess.get_mAP()
        self.log("val_mAP", mAP)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.args.lr)
        return optimizer

    def train_dataloader(self):
        args = self.args
        train_dataset = NewImagerLoader(args.source_path, args.train_file, args.json_path,
                                     args.gt_path, stride=args.train_stride, transform=get_transform(True))
        if args.num_gpus > 1:
            params = {'sampler': DistributedSampler(train_dataset)}
        else:
            params = {'shuffle': True}
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            **params)
        return train_loader

    def val_dataloader(self):
        args = self.args
        val_dataset = NewImagerLoader(args.source_path, args.val_file, args.json_path, args.gt_path,
                                   stride=args.val_stride, mode='val', transform=get_transform(False))
        if args.num_gpus > 1:
            params = {'sampler': DistributedSampler(val_dataset)}
        else:
            params = {'shuffle': True}
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            **params)
        return val_loader