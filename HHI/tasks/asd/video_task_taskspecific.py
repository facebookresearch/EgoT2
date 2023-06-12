#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import torch
from pytorch_lightning.core import LightningModule
from dataset.asd.dataLoader import train_loader_2task as ASDTrainDataset2Task
from dataset.asd.dataLoader import val_loader_2task as ASDValDataset2Task
from models.asd.build import build_model
from utils.utils import load_parameters, freeze_params
from .loss import lossAV, lossA, lossV


class ActiveSpeakerDetection2Loader(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.file_path = "/private/home/sherryxue/projects/TalkNet/data/"
        self.args = args
        self.checkpoint_metric = "val_acc"
        self.model = build_model(args)
        self.lossAV = lossAV(self.model.output_dim)

    def training_step(self, batch, batch_idx):
        audioFeature, asdvisualFeature, ttmvisualFeature, labels = batch
        if len(audioFeature.shape) == 1:
            print('skipping this batch...')
            return
        outsAV = self.model(ttmvisualFeature[0], asdvisualFeature[0], audioFeature[0], audioFeature[0])
        labels = labels[0].reshape((-1))
        nloss, _, _, prec = self.lossAV.forward(outsAV, labels)
        self.log("train_loss", nloss, on_step=True, on_epoch=True)
        self.log("train_correct", prec, on_step=False, on_epoch=True, reduce_fx="sum")
        self.log("train_total", len(labels), on_step=False, on_epoch=True, reduce_fx="sum")
        return nloss

    def validation_step(self, batch, batch_idx):
        audioFeature, asdvisualFeature, ttmvisualFeature, labels = batch
        outsAV = self.model(ttmvisualFeature[0], asdvisualFeature[0], audioFeature[0], audioFeature[0])
        labels = labels[0].reshape((-1))
        val_loss, predScore, _, prec = self.lossAV.forward(outsAV, labels)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        self.log("val_correct", prec, reduce_fx='sum')
        self.log("val_total", len(labels), reduce_fx='sum')
        return {
            'correct': prec,
            'total': len(labels)
        }

    def validation_epoch_end(self, outputs):
        num_correct = torch.tensor([x['correct'] for x in outputs]).sum()
        num_total = torch.tensor([x['total'] for x in outputs]).sum()
        self.log("val_acc", num_correct / num_total)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay), "interval": "step"}
        if self.args.nodecay:
            return optimizer
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = ASDTrainDataset2Task(trialFileName=os.path.join(self.file_path, "ego4d/csv/active_speaker_train.csv"),
                                  audioPath=os.path.join(self.file_path, "wave"),
                                  visualPath=os.path.join(self.file_path, "video_imgs"),
                                  batchSize=self.args.batch_size)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=self.args.num_workers)
        return loader

    def val_dataloader(self):
        dataset = ASDValDataset2Task(trialFileName=os.path.join(self.file_path, "ego4d/csv/active_speaker_val.csv"),
                                audioPath=os.path.join(self.file_path, "wave"),
                                visualPath=os.path.join(self.file_path, "video_imgs"))
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers)
        return loader
