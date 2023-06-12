#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import copy
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer.supporters import CombinedLoader
import optimizers.pnr.optimizer as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from dataset.pnr import loader
from dataset.lta import loader as lta_loader
from dataset.pnr.StateChangeDetectionAndKeyframeLocalisation import PNRDatasetwithAuxTaskSequenceLabel
from dataset.lta.long_term_anticipation_auxtask import Ego4dRecognitionwithAuxTaskSequenceLabel, Ego4dRecognitionwithAuxTaskSeparateSequenceLabel
from dataset.lta.long_term_anticipation_lta_auxtask import Ego4dLongTermAnticipationwithAuxTaskSeparateSequenceLabel
from models.multitask.video_model_builder import TaskPromptTransformer, TaskTranslationPromptTransformer, TaskTranslationPromptTransformer6Task
from evaluation.pnr.metrics import PnrOsccMetric, PNRMetric, OSCCMetric
from evaluation.lta.lta_metrics import ActionMetric, ARMetric, LTAMetric


class Unified3Task(LightningModule):
    checkpoint_metric = "val_loss"
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.save_hyperparameters()
        self.model = TaskPromptTransformer(args, vocab)  # task-specific model + unified sequence decoder
        self.criterion = nn.CrossEntropyLoss()
        self.metric = PnrOsccMetric(vocab)
        self.metric_ac = ActionMetric(vocab)
        self.pnrmetric = PNRMetric(vocab)
        self.osccmetric = OSCCMetric(vocab)
        self.armetric = ARMetric(vocab)

    def training_step(self, batch, batch_idx):
        frames_pnr, target_pnr, *_ = batch['pnr']
        logits_pnr = self.model(frames_pnr, target_pnr[:, :-1], 'pnr')
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])

        frames_oscc, _, target_oscc, *_ = batch['oscc']
        logits_oscc = self.model(frames_oscc, target_oscc[:, :-1], 'oscc')
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])

        frames_ac, target_ac, *_ = batch['action']
        logits_action = self.model(frames_ac, target_ac[:, :-1], 'action')
        loss3 = self.criterion(logits_action, target_ac[:, 1:])

        loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2 + self.args.ratio3 * loss3
        self.log("train_loss_pnr", loss1.item(), on_epoch=True, sync_dist=True)
        self.log("train_loss_oscc", loss2.item(), on_epoch=True, sync_dist=True)
        self.log("train_loss_ac", loss3.item(), on_epoch=True, sync_dist=True)
        self.log("train_loss", loss.item(), on_epoch=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        frames_pnr, target_pnr, _, fps, info, labels, _ = batch['pnr']
        x = frames_pnr.copy()
        logits_pnr = self.model(frames_pnr, target_pnr[:, :-1], 'pnr')
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])
        pred_pnr = self.model.predict(x, 'pnr')
        dist, err = self.metric.pnr_distance(pred_pnr, labels, fps, info)
        self.log("val_loss_pnr", loss1.item(), on_epoch=True, batch_size=self.val_batch_size1, sync_dist=True)
        self.log("val_pnr_err", err, on_epoch=True, batch_size=self.val_batch_size1, sync_dist=True)
        # self.log("val_pnr_dist", dist, on_epoch=True, batch_size=self.val_batch_size1, sync_dist=True)
        if batch_idx < self.loader1_stop_idx:
            self.pnrmetric.update(pred_pnr, labels, fps, info)
            self.log("val_pnr_dist_correct", dist, on_epoch=True, batch_size=self.val_batch_size1, sync_dist=True)

        frames_oscc, _, target_oscc, fps, info, _, state_change_label = batch['oscc']
        x = frames_oscc.copy()
        logits_oscc = self.model(frames_oscc, target_oscc[:, :-1], 'oscc')
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])
        pred_oscc = self.model.predict(x, 'oscc')
        acc2, err2 = self.metric.oscc_acc(pred_oscc, state_change_label)
        self.log("val_loss_oscc", loss2.item(), on_epoch=True, batch_size=self.val_batch_size2, sync_dist=True)
        self.log("val_oscc_err", err2, on_epoch=True, batch_size=self.val_batch_size2, sync_dist=True)
        # self.log("val_oscc_acc", acc2, on_epoch=True, batch_size=self.val_batch_size2, sync_dist=True)
        if batch_idx < self.loader2_stop_idx:
            self.osccmetric.update(pred_oscc, state_change_label)
            self.log("val_oscc_acc_correct", acc2, on_epoch=True, batch_size=self.val_batch_size2, sync_dist=True)

        frames_ac, target_ac, _, _, labels_ac = batch['action']
        x = frames_ac.copy()
        logits_action = self.model(frames_ac, target_ac[:, :-1], 'action')
        loss3 = self.criterion(logits_action, target_ac[:, 1:])
        pred_ac = self.model.predict_ac(x)
        vacc, nacc, verr, nerr = self.metric_ac.top1_error(pred_ac, labels_ac)
        self.log("val_loss_ac", loss3.item(), on_epoch=True, batch_size=self.val_batch_size3, sync_dist=True)
        self.log("val_ac_verr", verr, on_epoch=True, batch_size=self.val_batch_size3, sync_dist=True)
        self.log("val_ac_nerr", nerr, on_epoch=True, batch_size=self.val_batch_size3, sync_dist=True)
        self.log("val_ac_vacc", vacc, on_epoch=True, batch_size=self.val_batch_size3, sync_dist=True)
        self.log("val_ac_nacc", nacc, on_epoch=True, batch_size=self.val_batch_size3, sync_dist=True)

        self.armetric.update(pred_ac, labels_ac)

        loss = loss1 + loss2 + loss3
        self.log("val_loss", loss.item(), on_epoch=True)

    def validation_epoch_end(self, outputs):
        err1, pnr_dist = self.pnrmetric.compute()
        self.log('val_pnr_err_ddp', err1)
        self.log('val_pnr_dist_ddp', pnr_dist)
        self.pnrmetric.reset()

        err2, oscc_acc = self.osccmetric.compute()
        self.log('val_oscc_err_ddp', err2)
        self.log('val_oscc_acc_ddp', oscc_acc)
        self.osccmetric.reset()

        v_err, n_err, v_acc, n_acc = self.armetric.compute()
        self.log('val_ac_verr_ddp', v_err)
        self.log('val_ac_nerr_ddp', n_err)
        self.log('val_ac_vacc_ddp', v_acc)
        self.log('val_ac_nacc_ddp', n_acc)
        self.armetric.reset()


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        return optimizer

    def _construct_combined_loader(self, mode):
        train_loader1 = loader.construct_loader(self.model.cfg_pnr, mode, self.vocab)
        train_loader2 = loader.construct_loader(self.model.cfg_oscc, mode, self.vocab)
        train_loader3 = lta_loader.construct_loader(self.model.cfg_action, mode, self.vocab)
        loaders = {"pnr": train_loader1, "oscc": train_loader2, "action": train_loader3}
        combinedloader = CombinedLoader(loaders, mode="max_size_cycle")
        print(f"Loader 1 {len(train_loader1)} | Loader 2 {len(train_loader2)} | Loader 3 {len(train_loader3)}")
        if mode == "val":
            # self.args.num_gpus = 1
            self.loader1_stop_idx = len(train_loader1) / self.args.num_gpus
            self.loader2_stop_idx = len(train_loader2) / self.args.num_gpus
            self.loader3_stop_idx = len(train_loader3) / self.args.num_gpus
            print(f"Loader 1,2 stop idx {self.loader1_stop_idx}, {self.loader2_stop_idx}")

        print(f"{mode} loader (combined) {len(combinedloader)}")
        self.val_batch_size1 = self.model.cfg_pnr.TRAIN.BATCH_SIZE
        self.val_batch_size2 = self.model.cfg_oscc.TRAIN.BATCH_SIZE
        self.val_batch_size3 = self.model.cfg_action.TRAIN.BATCH_SIZE
        return combinedloader

    def setup(self, stage):
        self.train_loader = self._construct_combined_loader("train")
        self.val_loader = self._construct_combined_loader("val")
        self.test_loader = self._construct_combined_loader("val")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader



class Unified3TaskTranslation(LightningModule):
    checkpoint_metric = "val_loss"
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.save_hyperparameters()
        self.model = TaskTranslationPromptTransformer(args, vocab)  # task-specific model + unified sequence decoder
        self.criterion = nn.CrossEntropyLoss()
        self.pnrmetric = PNRMetric(vocab)
        self.osccmetric = OSCCMetric(vocab)
        self.armetric = ARMetric(vocab)

    def training_step(self, batch, batch_idx):
        frames_pnr, target_pnr, *_ = batch['pnr']['orig']
        frames_pnr_aux = batch['pnr']['recognition']
        logits_pnr = self.model(frames_pnr, frames_pnr_aux, target_pnr[:, :-1])
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])

        frames_oscc, _, target_oscc, *_ = batch['oscc']['orig']
        frames_oscc_aux = batch['oscc']['recognition']
        logits_oscc = self.model(frames_oscc, frames_oscc_aux, target_oscc[:, :-1])
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])

        frames_ac, target_ac, _, _, labels_ac = batch['action']['orig']
        frames_ac_aux = batch['action']['pnr']
        logits_ac = self.model(frames_ac_aux, frames_ac, target_ac[:, :-1])
        loss3 = self.criterion(logits_ac, target_ac[:, 1:])

        loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2 + self.args.ratio3 * loss3
        self.log("train_loss_pnr", loss1.item(), on_epoch=True)
        self.log("train_loss_oscc", loss2.item(), on_epoch=True)
        self.log("train_loss_ac", loss3.item(), on_epoch=True)
        self.log("train_loss", loss.item(), on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        frames_pnr, target_pnr, _, fps, info, labels, _ = batch['pnr']['orig']
        frames_pnr_aux = batch['pnr']['recognition']
        x1, x2 = frames_pnr.copy(), frames_pnr_aux.copy()
        logits_pnr = self.model(frames_pnr, frames_pnr_aux, target_pnr[:, :-1])
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])
        self.log("val_loss_pnr", loss1.item(), on_epoch=True)

        pred_pnr = self.model.predict(x1, x2, 'pnr')
        if batch_idx < self.loader1_stop_idx:
            self.pnrmetric.update(pred_pnr, labels, fps, info)

        frames_oscc, _, target_oscc, fps, info, _, state_change_label = batch['oscc']['orig']
        frames_oscc_aux = batch['oscc']['recognition']
        x3, x4 = frames_oscc.copy(), frames_oscc_aux.copy()
        logits_oscc = self.model(frames_oscc, frames_oscc_aux, target_oscc[:, :-1])
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])
        self.log("val_loss_oscc", loss2.item(), on_epoch=True)

        pred_oscc = self.model.predict(x3, x4, 'oscc')
        if batch_idx < self.loader2_stop_idx:
            self.osccmetric.update(pred_oscc, state_change_label)

        frames_ac, target_ac, _, _, labels_ac = batch['action']['orig']
        frames_ac_aux = batch['action']['pnr']
        x5, x6 = frames_ac_aux.copy(), frames_ac.copy()
        logits_action = self.model(frames_ac_aux, frames_ac, target_ac[:, :-1])
        loss3 = self.criterion(logits_action, target_ac[:, 1:])
        self.log("val_loss_ac", loss3.item(), on_epoch=True)

        pred_ac = self.model.predict_ac(x5, x6)  # (bs, 2)
        self.armetric.update(pred_ac, labels_ac)

        loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2 + self.args.ratio3 * loss3
        self.log("val_loss", loss.item(), on_epoch=True)

    def validation_epoch_end(self, outputs):
        err1, pnr_dist = self.pnrmetric.compute()
        self.log('val_pnr_err', err1)
        self.log('val_pnr_dist', pnr_dist)
        self.pnrmetric.reset()

        err2, oscc_acc = self.osccmetric.compute()
        self.log('val_oscc_err', err2)
        self.log('val_oscc_acc', oscc_acc)
        self.osccmetric.reset()

        v_err, n_err, v_acc, n_acc = self.armetric.compute()
        self.log('val_ac_verr', v_err)
        self.log('val_ac_nerr', n_err)
        self.log('val_ac_vacc', v_acc)
        self.log('val_ac_nacc', n_acc)
        self.armetric.reset()


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        if self.args.optim == "default":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
            return optimizer

        elif self.args.optim == "pnr":
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

    def _construct_combined_loader(self, mode):
        dataset1 = PNRDatasetwithAuxTaskSequenceLabel(self.model.cfg_pnr, self.model.cfg_action, self.vocab, mode)
        dataset2 = PNRDatasetwithAuxTaskSequenceLabel(self.model.cfg_oscc, self.model.cfg_action, self.vocab, mode)
        dataset3 = Ego4dRecognitionwithAuxTaskSequenceLabel(self.model.cfg_action, self.model.cfg_pnr, self.vocab, mode)

        loader1 = self._construct_loader(dataset1, mode, self.args.batch_size)
        loader2 = self._construct_loader(dataset2, mode, self.args.batch_size*2)
        loader3 = self._construct_loader(dataset3, mode, self.args.batch_size)
        loaders = {"pnr": loader1, "oscc": loader2, "action": loader3}

        # combinedloader = CombinedLoader(loaders, mode="min_size")
        combinedloader = CombinedLoader(loaders, mode="max_size_cycle")
        print(f"Loader 1 {len(loader1)} | Loader 2 {len(loader2)} | Loader 3 {len(loader3)}")
        if mode == "val":
            # self.args.num_gpus = 1
            self.loader1_stop_idx = len(loader1) / self.args.num_gpus
            self.loader2_stop_idx = len(loader2) / self.args.num_gpus
            self.loader3_stop_idx = len(loader3) / self.args.num_gpus
            print(f"Loader 1,2 stop idx {self.loader1_stop_idx}, {self.loader2_stop_idx}")

        print(f"{mode} loader (combined) {len(combinedloader)}")
        return combinedloader

    def setup(self, stage):
        self.train_loader = self._construct_combined_loader("train")
        self.val_loader = self._construct_combined_loader("val")
        self.test_loader = self._construct_combined_loader("val")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader



class Unified4TaskTranslation(LightningModule):
    checkpoint_metric = "val_loss"
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.save_hyperparameters()
        self.model = TaskTranslationPromptTransformer(args, vocab)  # task-specific model + unified sequence decoder
        self.criterion = nn.CrossEntropyLoss()
        self.pnrmetric = PNRMetric(vocab)
        self.osccmetric = OSCCMetric(vocab)
        self.armetric = ARMetric(vocab)

    def training_step(self, batch, batch_idx):
        batch_pnr, batch_oscc, batch_action = batch['pnr'], batch['oscc'], batch['action']
        frames_pnr, target_pnr, *_ = batch_pnr['orig']
        frames_pnr_aux = batch_pnr['recognition']
        logits_pnr = self.model(frames_pnr, frames_pnr_aux, target_pnr[:, :-1])
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])

        frames_oscc, _, target_oscc, *_ = batch_oscc['orig']
        frames_oscc_aux = batch_oscc['recognition']
        logits_oscc = self.model(frames_oscc, frames_oscc_aux, target_oscc[:, :-1])
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])

        frames_ac, target_ac_verb, target_ac_noun, _, _, labels_ac = batch_action['orig']
        frames_ac_aux = batch_action['pnr']
        frames_ac_aux2, frames_ac2 = frames_ac_aux.copy(), frames_ac.copy()

        logits_ac_verb = self.model(frames_ac_aux, frames_ac, target_ac_verb[:, :-1])
        loss3 = self.criterion(logits_ac_verb, target_ac_verb[:, 1:])

        logits_ac_noun = self.model(frames_ac_aux2, frames_ac2, target_ac_noun[:, :-1])
        loss4 = self.criterion(logits_ac_noun, target_ac_noun[:, 1:])

        loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2 + self.args.ratio3 * loss3 + self.args.ratio4 * loss4
        self.log("train_loss_pnr", loss1.item(), on_epoch=True)
        self.log("train_loss_oscc", loss2.item(), on_epoch=True)
        self.log("train_loss_ac_verb", loss3.item(), on_epoch=True)
        self.log("train_loss_ac_noun", loss4.item(), on_epoch=True)
        self.log("train_loss", loss.item(), on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        batch_pnr, batch_oscc, batch_action = batch['pnr'], batch['oscc'], batch['action']
        frames_pnr, target_pnr, _, fps, info, labels, _ = batch_pnr['orig']
        frames_pnr_aux = batch_pnr['recognition']
        x1, x2 = frames_pnr.copy(), frames_pnr_aux.copy()
        logits_pnr = self.model(frames_pnr, frames_pnr_aux, target_pnr[:, :-1])
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])
        self.log("val_loss_pnr", loss1.item(), on_epoch=True)

        pred_pnr = self.model.predict(x1, x2, 'pnr')
        self.pnrmetric.update(pred_pnr, labels, fps, info)

        frames_oscc, _, target_oscc, fps, info_oscc, _, state_change_label = batch_oscc['orig']
        frames_oscc_aux = batch_oscc['recognition']
        x3, x4 = frames_oscc.copy(), frames_oscc_aux.copy()
        logits_oscc = self.model(frames_oscc, frames_oscc_aux, target_oscc[:, :-1])
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])
        self.log("val_loss_oscc", loss2.item(), on_epoch=True)

        pred_oscc = self.model.predict(x3, x4, 'oscc')
        self.osccmetric.update(pred_oscc, state_change_label, info_oscc)

        frames_ac, target_ac_verb, target_ac_noun, _, _, labels_ac = batch_action['orig']
        frames_ac_aux = batch_action['pnr']
        frames_ac_aux2, frames_ac2 = frames_ac_aux.copy(), frames_ac.copy()
        x5, x6 = frames_ac_aux.copy(), frames_ac.copy()
        x7, x8 = frames_ac_aux.copy(), frames_ac.copy()

        logits_ac_verb = self.model(frames_ac_aux, frames_ac, target_ac_verb[:, :-1])
        loss3 = self.criterion(logits_ac_verb, target_ac_verb[:, 1:])
        logits_ac_noun = self.model(frames_ac_aux2, frames_ac2, target_ac_noun[:, :-1])
        loss4 = self.criterion(logits_ac_noun, target_ac_noun[:, 1:])
        self.log("val_loss_ac_verb", loss3.item(), on_epoch=True)
        self.log("val_loss_ac_noun", loss4.item(), on_epoch=True)

        pred_ac_verb = self.model.predict(x5, x6, 'action_verb')
        pred_ac_noun = self.model.predict(x7, x8, 'action_noun')
        pred_ac = torch.stack((pred_ac_verb, pred_ac_noun), dim=1)
        self.armetric.update(pred_ac, labels_ac)

        loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2 + self.args.ratio3 * loss3 + self.args.ratio4 * loss4
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
        self.log('val_oscc_cnt', oscc_cnt)  #28348
        self.osccmetric.reset()

        v_err, n_err, v_acc, n_acc = self.armetric.compute()
        self.log('val_ac_verr', v_err)
        self.log('val_ac_nerr', n_err)
        self.log('val_ac_vacc', v_acc)
        self.log('val_ac_nacc', n_acc)
        self.armetric.reset()


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if self.args.optim == "default":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
            return optimizer

        elif self.args.optim == "pnr":
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

    def _construct_combined_loader(self, mode):
        dataset1 = PNRDatasetwithAuxTaskSequenceLabel(self.model.cfg_pnr, self.model.cfg_action, self.vocab, mode)
        dataset2 = PNRDatasetwithAuxTaskSequenceLabel(self.model.cfg_oscc, self.model.cfg_action, self.vocab, mode)
        dataset3 = Ego4dRecognitionwithAuxTaskSeparateSequenceLabel(self.model.cfg_action, self.model.cfg_pnr, self.vocab, mode)

        if mode == "val":
            self.pnrmetric._construct_unique_id_mapping(dataset1.package)
            self.osccmetric._construct_unique_id_mapping(dataset2.package)

        loader1 = self._construct_loader(dataset1, mode, self.args.batch_size)
        loader2 = self._construct_loader(dataset2, mode, self.args.batch_size*2)
        loader3 = self._construct_loader(dataset3, mode, self.args.batch_size)
        loaders = {"pnr": loader1, "oscc": loader2, "action": loader3}

        combinedloader = CombinedLoader(loaders, mode=self.args.loader_mode)
        print(f"Loader 1 {len(loader1)} | Loader 2 {len(loader2)} | Loader 3 {len(loader3)}")

        print(f"{mode} loader (combined) {len(combinedloader)}")
        return combinedloader

    def setup(self, stage):
        self.train_loader = self._construct_combined_loader("train")
        self.val_loader = self._construct_combined_loader("val")
        self.test_loader = self._construct_combined_loader("val")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader



class Unified6TaskTranslationNoPredict(LightningModule):
    checkpoint_metric = "val_loss"
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.save_hyperparameters()
        self.model = TaskTranslationPromptTransformer6Task(args, vocab)  # task-specific model + unified sequence decoder
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        batch_pnr, batch_oscc, batch_action, batch_lta = batch['pnr'], batch['oscc'], batch['action'], batch['lta']
        frames_pnr, target_pnr, *_ = batch_pnr['orig']
        frames_pnr_aux = batch_pnr['recognition']
        logits_pnr = self.model(frames_pnr, frames_pnr_aux, target_pnr[:, :-1], 'pnr')
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])

        frames_oscc, _, target_oscc, *_ = batch_oscc['orig']
        frames_oscc_aux = batch_oscc['recognition']
        logits_oscc = self.model(frames_oscc, frames_oscc_aux, target_oscc[:, :-1], 'oscc')
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])

        frames_ac, target_ac_verb, target_ac_noun, _, _, labels_ac = batch_action['orig']
        frames_ac_aux = batch_action['pnr']
        frames_ac_aux2, frames_ac2 = frames_ac_aux.copy(), frames_ac.copy()

        logits_ac_verb = self.model(frames_ac_aux, frames_ac, target_ac_verb[:, :-1], 'action_verb')
        loss3 = self.criterion(logits_ac_verb, target_ac_verb[:, 1:])
        logits_ac_noun = self.model(frames_ac_aux2, frames_ac2, target_ac_noun[:, :-1], 'action_noun')
        loss4 = self.criterion(logits_ac_noun, target_ac_noun[:, 1:])

        frames_lta, target_lta_verb, target_lta_noun, *_ = batch_lta['orig']
        frames_lta_aux = batch_lta['pnr']
        frames_lta_aux2, frames_lta2 = copy.deepcopy(frames_lta_aux), frames_lta.copy()

        logits_lta_verb = self.model(frames_lta_aux, frames_lta, target_lta_verb[:, :-1], 'lta_verb')
        loss5 = self.criterion(logits_lta_verb, target_lta_verb[:, 1:])
        logits_lta_noun = self.model(frames_lta_aux2, frames_lta2, target_lta_noun[:, :-1], 'lta_noun')
        loss6 = self.criterion(logits_lta_noun, target_lta_noun[:, 1:])

        loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2 + self.args.ratio3 * loss3 + self.args.ratio4 * loss4 \
               + self.args.ratio5 * loss5 + self.args.ratio6 * loss6

        self.log("train_loss_pnr", loss1.item(), on_epoch=True)
        self.log("train_loss_oscc", loss2.item(), on_epoch=True)
        self.log("train_loss_ac_verb", loss3.item(), on_epoch=True)
        self.log("train_loss_ac_noun", loss4.item(), on_epoch=True)
        self.log("train_loss_lta_verb", loss5.item(), on_epoch=True)
        self.log("train_loss_lta_noun", loss6.item(), on_epoch=True)
        self.log("train_loss", loss.item(), on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        batch_pnr, batch_oscc, batch_action, batch_lta = batch['pnr'], batch['oscc'], batch['action'], batch['lta']

        frames_pnr, target_pnr, _, fps, info, labels, _ = batch_pnr['orig']
        frames_pnr_aux = batch_pnr['recognition']
        logits_pnr = self.model(frames_pnr, frames_pnr_aux, target_pnr[:, :-1], 'pnr')
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])

        frames_oscc, _, target_oscc, fps, info_oscc, _, state_change_label = batch_oscc['orig']
        frames_oscc_aux = batch_oscc['recognition']
        logits_oscc = self.model(frames_oscc, frames_oscc_aux, target_oscc[:, :-1], 'oscc')
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])

        frames_ac, target_ac_verb, target_ac_noun, _, _, labels_ac = batch_action['orig']
        frames_ac_aux = batch_action['pnr']
        frames_ac_aux2, frames_ac2 = frames_ac_aux.copy(), frames_ac.copy()
        logits_ac_verb = self.model(frames_ac_aux, frames_ac, target_ac_verb[:, :-1], 'action_verb')
        loss3 = self.criterion(logits_ac_verb, target_ac_verb[:, 1:])
        logits_ac_noun = self.model(frames_ac_aux2, frames_ac2, target_ac_noun[:, :-1], 'action_noun')
        loss4 = self.criterion(logits_ac_noun, target_ac_noun[:, 1:])

        frames_lta, target_lta_verb, target_lta_noun, _, unique_ids, _, _, forecast_labels = batch_lta['orig']
        frames_lta_aux = batch_lta['pnr']
        frames_lta_aux2, frames_lta2 = copy.deepcopy(frames_lta_aux), frames_lta.copy()
        logits_lta_verb = self.model(frames_lta_aux, frames_lta, target_lta_verb[:, :-1], 'lta_verb')
        loss5 = self.criterion(logits_lta_verb, target_lta_verb[:, 1:])
        logits_lta_noun = self.model(frames_lta_aux2, frames_lta2, target_lta_noun[:, :-1], 'lta_noun')
        loss6 = self.criterion(logits_lta_noun, target_lta_noun[:, 1:])

        loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2 + self.args.ratio3 * loss3 + self.args.ratio4 * loss4 \
               + self.args.ratio5 * loss5 + self.args.ratio6 * loss6

        self.log("val_loss_pnr", loss1.item(), on_epoch=True)
        self.log("val_loss_oscc", loss2.item(), on_epoch=True)
        self.log("val_loss_ac_verb", loss3.item(), on_epoch=True)
        self.log("val_loss_ac_noun", loss4.item(), on_epoch=True)
        self.log("val_loss_lta_verb", loss5.item(), on_epoch=True)
        self.log("val_loss_lta_noun", loss6.item(), on_epoch=True)
        self.log("val_loss", loss.item(), on_epoch=True)


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
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

    def _construct_combined_loader(self, mode):
        dataset1 = PNRDatasetwithAuxTaskSequenceLabel(self.model.cfg_pnr, self.model.cfg_action, self.vocab, mode)
        dataset2 = PNRDatasetwithAuxTaskSequenceLabel(self.model.cfg_oscc, self.model.cfg_action, self.vocab, mode)
        dataset3 = Ego4dRecognitionwithAuxTaskSeparateSequenceLabel(self.model.cfg_action, self.model.cfg_pnr, self.vocab, mode)
        dataset4 = Ego4dLongTermAnticipationwithAuxTaskSeparateSequenceLabel(self.model.cfg_lta, self.model.cfg_pnr, self.vocab, mode)

        loader1 = self._construct_loader(dataset1, mode, self.args.batch_size)
        loader2 = self._construct_loader(dataset2, mode, self.args.batch_size*2)
        loader3 = self._construct_loader(dataset3, mode, self.args.batch_size)
        loader4 = self._construct_loader(dataset4, mode, self.args.batch_size)
        loaders = {"pnr": loader1, "oscc": loader2, "action": loader3, "lta": loader4}

        combinedloader = CombinedLoader(loaders, mode=self.args.loader_mode)
        print(f'{mode} data | Loader len {len(loader1)} | {len(loader2)} | {len(loader3)} | {len(loader4)} | {len(combinedloader)}')
        return combinedloader

    def setup(self, stage):
        self.train_loader = self._construct_combined_loader("train")
        self.val_loader = self._construct_combined_loader("val")
        self.test_loader = self._construct_combined_loader("val")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader



class Unified6TaskTranslation(Unified6TaskTranslationNoPredict):
    checkpoint_metric = "val_loss"
    def __init__(self, args, vocab):
        super(Unified6TaskTranslation, self).__init__(args, vocab)
        self.pnrmetric = PNRMetric(vocab)
        self.osccmetric = OSCCMetric(vocab)
        self.armetric = ARMetric(vocab)
        self.ltametric = LTAMetric(vocab)

    def validation_step(self, batch, batch_idx):
        batch_pnr, batch_oscc, batch_action, batch_lta = batch['pnr'], batch['oscc'], batch['action'], batch['lta']
        frames_pnr, target_pnr, _, fps, info, labels, _ = batch_pnr['orig']
        frames_pnr_aux = batch_pnr['recognition']
        x1, x2 = frames_pnr.copy(), frames_pnr_aux.copy()

        logits_pnr = self.model(frames_pnr, frames_pnr_aux, target_pnr[:, :-1], 'pnr')
        loss1 = self.criterion(logits_pnr, target_pnr[:, 1:])
        pred_pnr = self.model.predict(x1, x2, 'pnr')
        self.pnrmetric.update(pred_pnr, labels, fps, info)

        frames_oscc, _, target_oscc, fps, info_oscc, _, state_change_label = batch_oscc['orig']
        frames_oscc_aux = batch_oscc['recognition']
        x3, x4 = frames_oscc.copy(), frames_oscc_aux.copy()

        logits_oscc = self.model(frames_oscc, frames_oscc_aux, target_oscc[:, :-1], 'oscc')
        loss2 = self.criterion(logits_oscc, target_oscc[:, 1:])
        pred_oscc = self.model.predict(x3, x4, 'oscc')
        self.osccmetric.update(pred_oscc, state_change_label, info_oscc)

        frames_ac, target_ac_verb, target_ac_noun, _, _, labels_ac = batch_action['orig']
        frames_ac_aux = batch_action['pnr']
        frames_ac_aux2, frames_ac2 = frames_ac_aux.copy(), frames_ac.copy()
        x5, x6 = frames_ac_aux.copy(), frames_ac.copy()

        logits_ac_verb = self.model(frames_ac_aux, frames_ac, target_ac_verb[:, :-1], 'action_verb')
        loss3 = self.criterion(logits_ac_verb, target_ac_verb[:, 1:])
        logits_ac_noun = self.model(frames_ac_aux2, frames_ac2, target_ac_noun[:, :-1], 'action_noun')
        loss4 = self.criterion(logits_ac_noun, target_ac_noun[:, 1:])
        pred_ac = self.model.predict(x5, x6, 'action')
        self.armetric.update(pred_ac, labels_ac)

        frames_lta, target_lta_verb, target_lta_noun, _, unique_ids, _, _, forecast_labels = batch_lta['orig']
        frames_lta_aux = batch_lta['pnr']
        frames_lta_aux2, frames_lta2 = copy.deepcopy(frames_lta_aux), frames_lta.copy()
        x7, x8 = copy.deepcopy(frames_lta_aux), frames_lta.copy()

        logits_lta_verb = self.model(frames_lta_aux, frames_lta, target_lta_verb[:, :-1], 'lta_verb')
        loss5 = self.criterion(logits_lta_verb, target_lta_verb[:, 1:])
        logits_lta_noun = self.model(frames_lta_aux2, frames_lta2, target_lta_noun[:, :-1], 'lta_noun')
        loss6 = self.criterion(logits_lta_noun, target_lta_noun[:, 1:])
        pred_lta = self.model.predict(x7, x8, 'lta')
        self.ltametric.update(pred_lta, forecast_labels.squeeze(1), unique_ids)

        loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2 + self.args.ratio3 * loss3 + self.args.ratio4 * loss4 \
               + self.args.ratio5 * loss5 + self.args.ratio6 * loss6

        self.log("val_loss_pnr", loss1.item(), on_epoch=True)
        self.log("val_loss_oscc", loss2.item(), on_epoch=True)
        self.log("val_loss_ac_verb", loss3.item(), on_epoch=True)
        self.log("val_loss_ac_noun", loss4.item(), on_epoch=True)
        self.log("val_loss_lta_verb", loss5.item(), on_epoch=True)
        self.log("val_loss_lta_noun", loss6.item(), on_epoch=True)
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
        self.log('val_oscc_cnt', oscc_cnt)  #28348
        self.osccmetric.reset()

        v_err, n_err, v_acc, n_acc = self.armetric.compute()
        self.log('val_ac_verr', v_err)
        self.log('val_ac_nerr', n_err)
        self.log('val_ac_vacc', v_acc)
        self.log('val_ac_nacc', n_acc)
        self.armetric.reset()

        v_err, v_acc, n_err, n_acc, lta_cnt = self.ltametric.compute()
        self.log('val_lta_verr', v_err)
        self.log('val_lta_nerr', n_err)
        self.log('val_lta_vacc', v_acc)
        self.log('val_lta_nacc', n_acc)
        self.log('val_lta_cnt', lta_cnt)
        self.ltametric.reset()

    def _construct_combined_loader(self, mode):
        dataset1 = PNRDatasetwithAuxTaskSequenceLabel(self.model.cfg_pnr, self.model.cfg_action, self.vocab, mode)
        dataset2 = PNRDatasetwithAuxTaskSequenceLabel(self.model.cfg_oscc, self.model.cfg_action, self.vocab, mode)
        dataset3 = Ego4dRecognitionwithAuxTaskSeparateSequenceLabel(self.model.cfg_action, self.model.cfg_pnr, self.vocab, mode)
        dataset4 = Ego4dLongTermAnticipationwithAuxTaskSeparateSequenceLabel(self.model.cfg_lta, self.model.cfg_pnr, self.vocab, mode)

        loader1 = self._construct_loader(dataset1, mode, self.args.batch_size)
        loader2 = self._construct_loader(dataset2, mode, self.args.batch_size*2)
        loader3 = self._construct_loader(dataset3, mode, self.args.batch_size)
        loader4 = self._construct_loader(dataset4, mode, self.args.batch_size)
        loaders = {"pnr": loader1, "oscc": loader2, "action": loader3, "lta": loader4}

        if mode == "val":
            self.pnrmetric._construct_unique_id_mapping(dataset1.package)
            self.osccmetric._construct_unique_id_mapping(dataset2.package)
            self.ltametric._construct_unique_id_mapping(dataset4.clip_annotations)

        combinedloader = CombinedLoader(loaders, mode=self.args.loader_mode)
        print(f'{mode} data | Loader len {len(loader1)} | {len(loader2)} | {len(loader3)} | {len(loader4)} | {len(combinedloader)}')
        return combinedloader