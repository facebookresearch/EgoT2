#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from dataset.lam.data_loader import NewImagerSeqLoader as LAMSeqDataset
from dataset.ttm.data_loader import ImagerSeqLoader as TTMSeqDataset
from dataset.asd.dataLoader import train_seqloader as ASDSeqTrainDataset
from dataset.asd.dataLoader import val_seqloader as ASDSeqValDataset
from dataset.ttm.sampler import SequenceBatchSampler
from models.multitask.task_prompt_model import TaskPromptTransformer
from utils.lam.utils import PostProcessor as LAMPostProcessor
from utils.lam.utils import get_transform as get_transform_lam
from utils.ttm.utils import PostProcessor as TTMPostProcessor
from utils.ttm.utils import get_transform as get_transform_ttm
from utils.ttm.utils import collate_fn_prompt
from utils.utils import build_vocab


class Unified3Task(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab = build_vocab()
        self.save_hyperparameters()
        self.checkpoint_metric = "val_loss"
        self.model = TaskPromptTransformer(args, self.vocab)
        self.criterion = nn.CrossEntropyLoss()
        self.asd_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        video_lam, target_lam = batch['lam']
        video_ttm, audio_ttm, target_ttm = batch['ttm']
        audio_asd, video_asd, target_asd = batch['asd']

        if len(audio_asd.shape) == 1:
            print('skipping this batch...')
            return

        logits_lam = self.model(video_lam, target_lam[:, :-1], 'lam')  # (bs, vocab_size-7, seq_len-2)
        loss_lam = self.criterion(logits_lam, target_lam[:, 1:])  # (bs, 2)
        logits_ttm = self.model(video_ttm, target_ttm[:, :-1], 'ttm', audio_ttm)
        loss_ttm = self.criterion(logits_ttm, target_ttm[:, 1:])
        target_asd = target_asd[0].reshape(-1, 3)
        logits_asd = self.model(video_asd[0], target_asd[:, :-1], 'asd', audio_asd[0])
        loss_asd = self.criterion(logits_asd, target_asd[:, 1:])

        loss = self.args.ratio1 * loss_lam + self.args.ratio2 * loss_ttm + self.args.ratio3 * loss_asd
        self.log("train_loss_lam", loss_lam, on_step=True, on_epoch=True)
        self.log("train_loss_ttm", loss_ttm, on_step=True, on_epoch=True)
        self.log("train_loss_asd", loss_asd, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self.lam_postprocess = LAMPostProcessor(self.args)
        self.ttm_postprocess = TTMPostProcessor(self.args)

    def validation_step(self, batch, batch_idx):
        batch_lam, batch_ttm, batch_asd = batch['lam'], batch['ttm'], batch['asd']
        video_lam, target_lam, targetseq_lam = batch_lam[0], batch_lam[1][0], batch_lam[1][1]
        video_ttm, audio_ttm, target_ttm, targetseq_ttm = batch_ttm[0], batch_ttm[1], batch_ttm[2][0], batch_ttm[2][1]
        audio_asd, video_asd, target_asd, targetseq_asd = batch_asd
        if len(audio_asd.shape) == 1:
            print('skipping this batch...')
            return

        logits_lam = self.model(video_lam, targetseq_lam[:, :-1], 'lam')
        loss_lam = self.criterion(logits_lam, targetseq_lam[:, 1:])
        output_lam = self.model.predict(video_lam, 'lam')

        if batch_idx < self.loader1_stop_idx:
            self.lam_postprocess.update(output_lam, target_lam)

        logits_ttm = self.model(video_ttm, targetseq_ttm[:, :-1], 'ttm', audio_ttm)
        loss_ttm = self.criterion(logits_ttm, targetseq_ttm[:, 1:])
        output_ttm = self.model.predict(video_ttm, 'ttm', audio_ttm)

        if batch_idx < self.loader2_stop_idx:
            self.ttm_postprocess.update(output_ttm, target_ttm)

        labels_asd = target_asd[0].reshape((-1))
        targetseq_asd = targetseq_asd[0].reshape(-1, 3)
        logits_asd = self.model(video_asd[0], targetseq_asd[:, :-1], 'asd', audio_asd[0])
        loss_asd = self.criterion(logits_asd, targetseq_asd[:, 1:])
        output_asd = self.model.predict(video_asd[0], 'asd', audio_asd[0])
        self.asd_acc.update(output_asd, labels_asd)

        loss = self.args.ratio1 * loss_lam + self.args.ratio2 * loss_ttm + self.args.ratio3 * loss_asd
        self.log("val_loss_lam", loss_lam)
        self.log("val_loss_ttm", loss_ttm)
        self.log("val_loss_asd", loss_asd)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        self.lam_postprocess.save()
        lam_mAP, lam_acc = self.lam_postprocess.get_mAP()
        self.log("val_lam_mAP", lam_mAP)
        self.log("val_lam_acc", lam_acc)

        self.ttm_postprocess.save()
        ttm_mAP, ttm_acc = self.ttm_postprocess.get_mAP()
        self.log("val_ttm_mAP", ttm_mAP)
        self.log("val_ttm_acc", ttm_acc)

        asd_acc = self.asd_acc.compute()
        self.log("val_asd_acc", asd_acc)
        self.asd_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        args = self.args
        lam_dataset = LAMSeqDataset(self.vocab, args.lam_source_path, args.lam_train_file, args.lam_json_path,
                                     args.lam_gt_path, stride=args.lam_train_stride, transform=get_transform_lam(True))
        lam_loader = torch.utils.data.DataLoader(
            lam_dataset,
            batch_size=args.lam_train_batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            shuffle=True)

        ttm_dataset = TTMSeqDataset(self.vocab, args.ttm_img_path, args.ttm_wave_path, args.ttm_train_file, args.ttm_json_path,
                                     args.ttm_gt_path, stride=args.ttm_train_stride, transform=get_transform_ttm(True))
        ttm_loader = torch.utils.data.DataLoader(
            ttm_dataset,
            batch_sampler=SequenceBatchSampler(ttm_dataset, args.ttm_batch_size),
            num_workers=args.num_workers,
            collate_fn=collate_fn_prompt,
            pin_memory=False)

        asd_dataset = ASDSeqTrainDataset(vocab=self.vocab,
                                    trialFileName=os.path.join(args.asd_file_path, "ego4d/csv/active_speaker_train.csv"),
                                    audioPath=os.path.join(args.asd_file_path, "wave"),
                                    visualPath=os.path.join(args.asd_file_path, "video_imgs"),
                                    batchSize=self.args.asd_batch_size)
        asd_loader = torch.utils.data.DataLoader(asd_dataset, batch_size=1, shuffle=True, num_workers=self.args.num_workers)
        loaders = {"lam": lam_loader, "ttm": ttm_loader, "asd": asd_loader}
        combinedloader = CombinedLoader(loaders, mode="min_size")
        print(f"loader len: LAM {len(lam_loader)} | TTM {len(ttm_loader)} | ASD {len(asd_loader)}")
        return combinedloader


    def val_dataloader(self):
        args = self.args
        lam_dataset = LAMSeqDataset(self.vocab, args.lam_source_path, args.lam_val_file, args.lam_json_path,
                                     args.lam_gt_path, stride=args.lam_val_stride, mode='val', transform=get_transform_lam(False))
        lam_loader = torch.utils.data.DataLoader(
            lam_dataset,
            batch_size=args.lam_val_batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            shuffle=True)

        ttm_dataset = TTMSeqDataset(self.vocab, args.ttm_img_path, args.ttm_wave_path, args.ttm_val_file, args.ttm_json_path,
                                     args.ttm_gt_path, stride=args.ttm_val_stride, mode='val', transform=get_transform_ttm(False))
        ttm_loader = torch.utils.data.DataLoader(
            ttm_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=False)

        asd_dataset = ASDSeqValDataset(vocab=self.vocab,
                                    trialFileName=os.path.join(args.asd_file_path, "ego4d/csv/active_speaker_val.csv"),
                                    audioPath=os.path.join(args.asd_file_path, "wave"),
                                    visualPath=os.path.join(args.asd_file_path, "video_imgs"),
                                    batchSize=self.args.asd_batch_size)
        asd_loader = torch.utils.data.DataLoader(asd_dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers)
        loaders = {"lam": lam_loader, "ttm": ttm_loader, "asd": asd_loader}
        combinedloader = CombinedLoader(loaders, mode="max_size_cycle")
        print(f"loader len: LAM {len(lam_loader)} | TTM {len(ttm_loader)} | ASD {len(asd_loader)}")
        self.loader1_stop_idx = len(lam_loader)
        self.loader2_stop_idx = len(ttm_loader)
        return combinedloader