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
from optimizers.lta import lr_scheduler
from dataset.lta.long_term_anticipation import Ego4dRecognitionSeparateSequenceLabel, Ego4dLongTermAnticipationSeparateSequenceLabel
from models.multitask.video_model_builder_action import TaskTranslationPromptTransformerActionTask, TaskTranslationPromptTransformerTemporalActionTask
from evaluation.lta import lta_metrics as metrics
from utils.lta.parser import load_config_from_file as load_lta_config


class Unified4TaskTranslationAction(LightningModule):
    checkpoint_metric = "val_loss"
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.save_hyperparameters()
        self.cfg_action = load_lta_config(self.args.action_cfg_file)
        self.cfg_lta = load_lta_config(self.args.lta_cfg_file)
        if args.model == "temporal":
            self.model = TaskTranslationPromptTransformerTemporalActionTask(args, vocab)
        else:
            self.model = TaskTranslationPromptTransformerActionTask(args, vocab)  # task-specific model + unified sequence decoder
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        for key in ['action', 'lta']:
            input, target_seq_verb, target_seq_noun, *_ = batch[key]
            input2 = input.copy()

            logits_verb = self.model(input, target_seq_verb[:, :-1], key+'_verb')  # (bs, 601, 2)
            loss1 = self.criterion(logits_verb, target_seq_verb[:, 1:])
            logits_noun = self.model(input2, target_seq_noun[:, :-1], key+'_noun')  # (bs, 601, 2)
            loss2 = self.criterion(logits_noun, target_seq_noun[:, 1:])

            self.log('train_loss_' + key + '_verb', loss1.item(), on_epoch=True)
            self.log('train_loss_' + key + '_noun', loss2.item(), on_epoch=True)

            if key == 'action':
                loss = self.args.ratio1 * loss1 + self.args.ratio2 * loss2
            else:
                loss += self.args.ratio3 * loss1 + self.args.ratio4 * loss2

        self.log("train_loss", loss.item(), on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input, target_seq_verb, target_seq_noun, *_, labels = batch['action']
        input2, input3 = input.copy(), input.copy()

        logits_verb = self.model(input, target_seq_verb[:, :-1], 'action_verb')
        loss1 = self.criterion(logits_verb, target_seq_verb[:, 1:])
        logits_noun = self.model(input2, target_seq_noun[:, :-1], 'action_noun')
        loss2 = self.criterion(logits_noun, target_seq_noun[:, 1:])
        self.log("val_loss_action_verb", loss1.item(), on_epoch=True)
        self.log("val_loss_action_noun", loss2.item(), on_epoch=True)

        preds = self.model.predict(input3, 'action')
        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5)
        )
        step_result = {
            "val_top1_verb_err": top1_err_verb.item(),
            "val_top5_verb_err": top5_err_verb.item(),
            "val_top1_noun_err": top1_err_noun.item(),
            "val_top5_noun_err": top5_err_noun.item(),
        }

        input, target_seq_verb, target_seq_noun, *_, label_clip_times, forecast_labels = batch['lta']
        input2, input3 = input.copy(), input.copy()
        logits_verb = self.model(input, target_seq_verb[:, :-1], 'lta_verb')
        loss3 = self.criterion(logits_verb, target_seq_verb[:, 1:])
        logits_noun = self.model(input2, target_seq_noun[:, :-1], 'lta_noun')
        loss4 = self.criterion(logits_noun, target_seq_noun[:, 1:])
        self.log("val_loss_lta_verb", loss3.item(), on_epoch=True)
        self.log("val_loss_lta_noun", loss4.item(), on_epoch=True)

        preds = self.model.generate(input3)

        for head_idx, pred in enumerate(preds):
            assert pred.shape[1] == self.model.k
            bi, ki, zi = (0, 1, 2)
            pred = pred.permute(bi, zi, ki)
            pred, forecast_labels = pred.cpu(), forecast_labels.cpu()

            label = forecast_labels[:, :, head_idx: head_idx + 1]
            auedit = metrics.distributed_AUED(pred, label)
            results = {
                f"val_{head_idx}_" + k: v for k, v in auedit.items()
            }
            step_result.update(results)

        loss =  self.args.ratio1 * loss1 + self.args.ratio2 * loss2 + self.args.ratio3 * loss3 + self.args.ratio4 * loss4
        self.log("val_loss", loss.item(), on_epoch=True)

        return step_result


    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if self.args.optim == "default":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.lr, weight_decay=1e-4
            )
            return optimizer

        elif self.args.optim == "lta":
            steps_in_epoch = len(self.train_loader)
            return lr_scheduler.lr_factory(
                self.model, self.cfg_lta, steps_in_epoch, self.cfg_lta.SOLVER.LR_POLICY
            )

        elif self.args.optim == "action":
            steps_in_epoch = len(self.train_loader)
            return lr_scheduler.lr_factory(
                self.model, self.cfg_action, steps_in_epoch, self.cfg_action.SOLVER.LR_POLICY
            )

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
        dataset1 = Ego4dRecognitionSeparateSequenceLabel(self.cfg_action, self.vocab, mode)
        dataset2 = Ego4dLongTermAnticipationSeparateSequenceLabel(self.cfg_lta, self.vocab, mode)

        loader1 = self._construct_loader(dataset1, mode, self.args.batch_size)
        loader2 = self._construct_loader(dataset2, mode, self.args.batch_size)
        loaders = {"action": loader1, "lta": loader2}

        # combinedloader = CombinedLoader(loaders, mode="min_size")
        combinedloader = CombinedLoader(loaders, mode=self.args.loader_mode)
        print(f"Loader 1 {len(loader1)} | Loader 2 {len(loader2)} |")
        if mode == "val":
            self.loader2_stop_idx = len(loader2) / self.args.num_gpus
            print(f"Loader 2 stop idx {self.loader2_stop_idx}")

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
