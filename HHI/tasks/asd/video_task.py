#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import torch
from pytorch_lightning.core import LightningModule
from dataset.asd.dataLoader import train_loader as ASDTrainDataset
from dataset.asd.dataLoader import val_loader as ASDValDataset
from models.asd.talkNetModel import talkNetModel
from utils.utils import load_ckpt, load_parameters, freeze_params
from .loss import lossAV, lossA, lossV


class ActiveSpeakerDetection(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.file_path = args.data_file_path
        self.args = args
        self.model = talkNetModel()
        self.lossAV = lossAV()
        self.lossA = lossA()
        self.lossV = lossV()
        self.checkpoint_metric = "val_acc"
        if args.init_from_ava:
            load_ckpt(self.model, args.checkpoint, load_asd=True)
            # load_parameters(self.state_dict(), "/private/home/sherryxue/projects/TalkNet/data/pretrain_AVA.model")
        if args.finetune:
            load_parameters(self.state_dict(), args.asd_checkpoint)
            freeze_params(self.model)

    def training_step(self, batch, batch_idx):
        audioFeature, visualFeature, labels = batch
        if len(audioFeature.shape) == 1:
            print('skipping this batch...')
            return
        audioEmbed = self.model.forward_audio_frontend(audioFeature[0])  #(bs, D, 128) # feedForward
        visualEmbed = self.model.forward_visual_frontend(visualFeature[0]) #(bs, D, 128)
        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed) #(bs*D, 256)
        outsA = self.model.forward_audio_backend(audioEmbed)  #(bs*D, 128)
        outsV = self.model.forward_visual_backend(visualEmbed)  #(bs*D, 128)
        labels = labels[0].reshape((-1))  # (bs*D,)
        nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
        nlossA = self.lossA.forward(outsA, labels)
        nlossV = self.lossV.forward(outsV, labels)
        nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
        self.log("train_loss", nloss, on_step=True, on_epoch=True)
        self.log("train_correct", prec, on_step=False, on_epoch=True, reduce_fx="sum")
        self.log("train_total", len(labels), on_step=False, on_epoch=True, reduce_fx="sum")
        return nloss

    def validation_step(self, batch, batch_idx):
        audioFeature, visualFeature, labels = batch
        audioEmbed = self.model.forward_audio_frontend(audioFeature[0])
        visualEmbed = self.model.forward_visual_frontend(visualFeature[0])
        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
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
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = ASDTrainDataset(trialFileName=os.path.join(self.file_path, "ego4d/csv/active_speaker_train.csv"),
                                  audioPath=os.path.join(self.file_path, "wave"),
                                  visualPath=os.path.join(self.file_path, "video_imgs"),
                                  batchSize=self.args.batch_size)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=self.args.num_workers)
        return loader

    def val_dataloader(self):
        dataset = ASDValDataset(trialFileName=os.path.join(self.file_path, "ego4d/csv/active_speaker_val.csv"),
                                audioPath=os.path.join(self.file_path, "wave"),
                                visualPath=os.path.join(self.file_path, "video_imgs"))
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers)
        return loader
