#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from dataset.pnr import loader
from dataset.lta import loader as lta_loader
from dataset.pnr.StateChangeDetectionAndKeyframeLocalisation import PNRDatasetSequenceLabel, PNRDatasetwithAuxTaskSequenceLabel
from dataset.lta.long_term_anticipation_auxtask import Ego4dRecognitionwithAuxTaskSequenceLabel
from optimizers.lta import lr_scheduler
import optimizers.pnr.optimizer as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from models.multitask.video_model_builder_2task import TaskTranslationPromptTransformer2Task
from evaluation.pnr.metrics import PNRMetric, OSCCMetric
from evaluation.lta.lta_metrics import  ARMetric


class PnrOnlyTaskTranslation(LightningModule):
    checkpoint_metric = "val_loss_pnr"

    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.save_hyperparameters()
        self.model = TaskTranslationPromptTransformerLearnPE(args, vocab)  # task-specific model + unified sequence decoder
        self.criterion = nn.CrossEntropyLoss()
        self.pnrmetric = PNRMetric(vocab)

    def training_step(self, batch, batch_idx):
        frames_pnr, target_pnr, *_ = batch['orig']
        frames_pnr_aux = batch['recognition']
        logits_pnr = self.model(frames_pnr, frames_pnr_aux, target_pnr[:, :-1])
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])
        self.log("train_loss_pnr", loss1.item(), on_epoch=True)
        return loss1

    def validation_step(self, batch, batch_idx):
        frames_pnr, target_pnr, _, fps, info, labels, _ = batch['orig']
        frames_pnr_aux = batch['recognition']
        x1, x2 = frames_pnr.copy(), frames_pnr_aux.copy()
        logits_pnr = self.model(frames_pnr, frames_pnr_aux, target_pnr[:, :-1])
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])
        self.log("val_loss_pnr", loss1.item(), on_epoch=True)

        pred_pnr = self.model.predict(x1, x2, 'pnr')
        self.pnrmetric.update(pred_pnr, labels, fps, info)

    def validation_epoch_end(self, outputs):
        err1, pnr_dist = self.pnrmetric.compute()
        self.log('val_pnr_err', err1)
        self.log('val_pnr_dist', pnr_dist)
        self.pnrmetric.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.construct_optimizer(self.model, self.model.cfg_pnr)
        steps_in_epoch = len(self.train_loader)
        if self.model.cfg_pnr.SOLVER.LR_POLICY == "cosine":
            slow_fast_scheduler = CosineAnnealingLR(
                optimizer, self.model.cfg_pnr.SOLVER.MAX_EPOCH * steps_in_epoch, last_epoch=-1
            )
        elif self.model.cfg_pnr.SOLVER.LR_POLICY == "constant":
            slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1)
        else:

            def lr_lambda(step):
                return optim.get_epoch_lr(step / steps_in_epoch, self.model.cfg_pnr)

            slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        scheduler = {"scheduler": slow_fast_scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def _construct_loader(self, dataset, mode, batch_size):
        shuffle = True if mode == "train" else False
        drop_last = True if mode == "train" else False
        sampler = DistributedSampler(dataset) if self.args.num_gpus > 1 else None
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )
        return loader

    def _construct_pnr_loader(self, mode):
        dataset1 = PNRDatasetwithAuxTaskSequenceLabel(self.model.cfg_pnr, self.model.cfg_action, self.vocab, mode)
        loader1 = self._construct_loader(dataset1, mode, self.args.batch_size)
        return loader1

    def setup(self, stage):
        self.train_loader = self._construct_pnr_loader("train")
        self.val_loader = self._construct_pnr_loader("val")
        self.test_loader = self._construct_pnr_loader("val")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class OsccOnlyTaskTranslation(LightningModule):
    checkpoint_metric = "val_loss_oscc"

    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.save_hyperparameters()
        self.model = TaskTranslationPromptTransformerLearnPE(args, vocab)  # task-specific model + unified sequence decoder
        self.criterion = nn.CrossEntropyLoss()
        self.osccmetric = OSCCMetric(vocab)

    def training_step(self, batch, batch_idx):
        frames_oscc, _, target_oscc, *_ = batch['orig']
        frames_oscc_aux = batch['recognition']
        logits_oscc = self.model(frames_oscc, frames_oscc_aux, target_oscc[:, :-1])
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])
        self.log("train_loss_oscc", loss2.item(), on_epoch=True)
        return loss2

    def validation_step(self, batch, batch_idx):
        frames_oscc, _, target_oscc, fps, info, _, state_change_label = batch['orig']
        frames_oscc_aux = batch['recognition']
        x3, x4 = frames_oscc.copy(), frames_oscc_aux.copy()
        logits_oscc = self.model(frames_oscc, frames_oscc_aux, target_oscc[:, :-1])
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])
        self.log("val_loss_oscc", loss2.item(), on_epoch=True)

        pred_oscc = self.model.predict(x3, x4, 'oscc')
        self.osccmetric.update(pred_oscc, state_change_label)

    def validation_epoch_end(self, outputs):
        err2, oscc_acc = self.osccmetric.compute()
        self.log('val_oscc_err', err2)
        self.log('val_oscc_acc', oscc_acc)
        self.osccmetric.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.construct_optimizer(self.model, self.model.cfg_pnr)
        steps_in_epoch = len(self.train_loader)
        if self.model.cfg_pnr.SOLVER.LR_POLICY == "cosine":
            slow_fast_scheduler = CosineAnnealingLR(
                optimizer, self.model.cfg_pnr.SOLVER.MAX_EPOCH * steps_in_epoch, last_epoch=-1
            )
        elif self.model.cfg_pnr.SOLVER.LR_POLICY == "constant":
            slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1)
        else:

            def lr_lambda(step):
                return optim.get_epoch_lr(step / steps_in_epoch, self.model.cfg_pnr)

            slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        scheduler = {"scheduler": slow_fast_scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def _construct_loader(self, dataset, mode, batch_size):
        shuffle = True if mode == "train" else False
        drop_last = True if mode == "train" else False
        sampler = DistributedSampler(dataset) if self.args.num_gpus > 1 else None
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )
        return loader

    def _construct_oscc_loader(self, mode):
        dataset2 = PNRDatasetwithAuxTaskSequenceLabel(self.model.cfg_oscc, self.model.cfg_action, self.vocab, mode)
        loader2 = self._construct_loader(dataset2, mode, self.args.batch_size)
        return loader2

    def setup(self, stage):
        self.train_loader = self._construct_oscc_loader("train")
        self.val_loader = self._construct_oscc_loader("val")
        self.test_loader = self._construct_oscc_loader("val")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class ActionOnlyTaskTranslation(LightningModule):
    checkpoint_metric = "val_loss_ac"
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.save_hyperparameters()
        self.model = TaskTranslationPromptTransformerLearnPE(args, vocab)  # task-specific model + unified sequence decoder
        self.criterion = nn.CrossEntropyLoss()
        self.pnrmetric = PNRMetric(vocab)
        self.osccmetric = OSCCMetric(vocab)
        self.armetric = ARMetric(vocab)

    def training_step(self, batch, batch_idx):
        frames_ac, target_ac, _, _, labels_ac = batch['orig']
        frames_ac_aux = batch['pnr']
        logits_ac = self.model(frames_ac_aux, frames_ac, target_ac[:, :-1])
        loss = self.criterion(logits_ac, target_ac[:, 1:])

        self.log("train_loss_ac", loss.item(), on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        frames_ac, target_ac, _, _, labels_ac = batch['orig']
        frames_ac_aux = batch['pnr']
        x5, x6 = frames_ac_aux.copy(), frames_ac.copy()
        logits_action = self.model(frames_ac_aux, frames_ac, target_ac[:, :-1])
        loss = self.criterion(logits_action, target_ac[:, 1:])
        self.log("val_loss_ac", loss.item(), on_epoch=True)

        pred_ac = self.model.predict_ac(x5, x6)
        self.armetric.update(pred_ac, labels_ac)


    def validation_epoch_end(self, outputs):
        v_err, n_err, v_acc, n_acc = self.armetric.compute()
        self.log('val_ac_verr', v_err)
        self.log('val_ac_nerr', n_err)
        self.log('val_ac_vacc', v_acc)
        self.log('val_ac_nacc', n_acc)
        self.armetric.reset()


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        return optimizer

    def _construct_loader(self, dataset, mode, batch_size):
        shuffle = True if mode == "train" else False
        drop_last = True if mode == "train" else False
        sampler = DistributedSampler(dataset) if self.args.num_gpus > 1 else None
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )
        return loader

    def _construct_ar_loader(self, mode):
        dataset3 = Ego4dRecognitionwithAuxTaskSequenceLabel(self.model.cfg_action, self.model.cfg_pnr, self.vocab, mode)
        loader3 = self._construct_loader(dataset3, mode, self.args.batch_size)
        return loader3

    def setup(self, stage):
        self.train_loader = self._construct_ar_loader("train")
        self.val_loader = self._construct_ar_loader("val")
        self.test_loader = self._construct_ar_loader("val")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


from utils.multitask.build_vocab import build_vocab_task12
class Task12Translation(LightningModule):
    checkpoint_metric = "val_loss"

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab = build_vocab_task12()
        self.save_hyperparameters()
        self.model = TaskTranslationPromptTransformer2Task(args, self.vocab)  # task-specific model + unified sequence decoder
        self.criterion = nn.CrossEntropyLoss()
        self.pnrmetric = PNRMetric(self.vocab)
        self.osccmetric = OSCCMetric(self.vocab)

    def training_step(self, batch, batch_idx):
        frames_pnr, target_pnr, *_ = batch['pnr']
        logits_pnr = self.model(frames_pnr, target_pnr[:, :-1])
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])

        frames_oscc, _, target_oscc, *_ = batch['oscc']
        logits_oscc = self.model(frames_oscc, target_oscc[:, :-1])
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])

        loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2
        self.log("train_loss_pnr", loss1.item(), on_epoch=True)
        self.log("train_loss_oscc", loss2.item(), on_epoch=True)
        self.log("train_loss", loss.item(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames_pnr, target_pnr, _, fps, info, labels, _ = batch['pnr']
        x = frames_pnr.copy()
        logits_pnr = self.model(frames_pnr, target_pnr[:, :-1])
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])
        self.log("val_loss_pnr", loss1.item(), on_epoch=True)

        pred_pnr = self.model.predict(x, 'pnr')
        self.pnrmetric.update(pred_pnr, labels, fps, info)

        frames_oscc, _, target_oscc, fps, info, _, state_change_label = batch['oscc']
        x = frames_oscc.copy()
        logits_oscc = self.model(frames_oscc, target_oscc[:, :-1])
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])

        pred_oscc = self.model.predict(x, 'oscc')
        self.osccmetric.update(pred_oscc, state_change_label, info)

        loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2
        self.log("val_loss", loss.item(), on_epoch=True)

    def validation_epoch_end(self, outputs):
        err1, pnr_dist, pnr_cnt = self.pnrmetric.compute()
        self.log('val_pnr_err', err1)
        self.log('val_pnr_dist', pnr_dist)
        self.log('val_pnr_cnt', pnr_cnt)
        self.pnrmetric.reset()

        err2, oscc_acc, oscc_cnt = self.osccmetric.compute()
        self.log('val_oscc_err', err2)
        self.log('val_oscc_acc', oscc_acc)
        self.log('val_oscc_cnt', oscc_cnt)  # 28348
        self.osccmetric.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd
        )
        return optimizer

    def _construct_loader(self, dataset, mode, batch_size):
        shuffle = True if mode == "train" else False
        drop_last = True if mode == "train" else False
        sampler = DistributedSampler(dataset) if self.args.num_gpus > 1 else None
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )
        return loader

    def _construct_task12_loader(self, mode):
        dataset1 = PNRDatasetSequenceLabel(self.model.cfg_pnr, self.vocab, mode)
        dataset2 = PNRDatasetSequenceLabel(self.model.cfg_oscc, self.vocab, mode)
        if mode  == "val":
            self.pnrmetric._construct_unique_id_mapping(dataset1.package)
            self.osccmetric._construct_unique_id_mapping(dataset2.package)
        loader1 = self._construct_loader(dataset1, mode, self.args.batch_size)
        loader2 = self._construct_loader(dataset2, mode, self.args.batch_size)
        loaders = {"pnr": loader1, "oscc": loader2}
        combinedloader = CombinedLoader(loaders, mode="max_size_cycle")
        print(f"Loader 1 {len(loader1)} | Loader 2 {len(loader2)}")
        print(f"{mode} loader (combined) {len(combinedloader)}")
        return combinedloader

    def setup(self, stage):
        self.train_loader = self._construct_task12_loader("train")
        self.val_loader = self._construct_task12_loader("val")
        self.test_loader = self._construct_task12_loader("val")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
