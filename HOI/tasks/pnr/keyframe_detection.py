"""
This file contains code for the task of keyframe detection
"""

#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
from torch.nn import MSELoss, BCELoss
import numpy as np
from tasks.pnr.video_task import VideoTask
from evaluation.pnr.metrics import state_change_accuracy, keyframe_distance, keyframe_accuracy

mse = MSELoss(reduction='mean')
bce = BCELoss(reduction='none')

class KeyframeLocalisation(VideoTask):
    checkpoint_metric = "keyframe_loc_metric_time_dist_val_neg"

    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        keyframe_preds = self.forward(frames)

        if self.cfg.MODEL.LOSS_FUNC == "bce":
            pred = torch.sigmoid(keyframe_preds.squeeze(dim=1))
            keyframe_loss = self.loss_fun(pred, labels.float())

        else:
            keyframe_preds_ = keyframe_preds.permute(0, 2, 1)  # (bs, 16, 1)

            keyframe_loss = self.loss_fun(
                keyframe_preds_.squeeze(dim=-1),  # (bs, 16)
                torch.argmax(labels.long(), dim=1)  # bs
            )
            # We want to calculate the keyframe loss only for the clips with state
            # change
            keyframe_loss = torch.mean(state_change_label.T * keyframe_loss)

        loss = keyframe_loss

        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_label,
            fps,
            info
        )

        correct, total = keyframe_accuracy(keyframe_preds.squeeze(1), labels, state_change_label)
        pl_acc = correct / total

        return {
            "keyframe_loss": keyframe_loss,
            "train_loss": loss,
            "loss": loss,
            "pseudo_acc": pl_acc,
            "keyframe_loc_metric_time_dist_train": keyframe_avg_time_dist
        }

    def training_epoch_end(self, training_step_outputs):
        keys = [x for x in training_step_outputs[0].keys()]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].mean() for item in training_step_outputs]
                )
            ).mean()
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        keyframe_preds = self.forward(frames)
        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_label,
            fps,
            info
        )
        correct, total = keyframe_accuracy(keyframe_preds.squeeze(1), labels, state_change_label)
        pl_acc = correct / total

        return {
            "keyframe_loc_metric_time_dist_val": keyframe_avg_time_dist,
            "keyframe_loc_metric_time_dist_val_neg": -keyframe_avg_time_dist,
            "pseudo_acc": pl_acc
        }

    def validation_epoch_end(self, validation_step_outputs):
        keys = [x for x in validation_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].mean() for item in validation_step_outputs]
                )
            ).mean()
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)


class KeyframeLocalisationCnnLSTM(KeyframeLocalisation):
    checkpoint_metric = "keyframe_loc_metric_time_dist_val_neg"

    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch

        keyframe_preds = self.forward(frames)
        keyframe_loss = self.loss_fun(keyframe_preds, labels.float())

        # We want to calculate the keyframe loss only for the clips with state
        # change
        # keyframe_loss = torch.mean(state_change_label.T * keyframe_loss)
        loss = keyframe_loss

        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_label,
            fps,
            info
        )

        return {
            "keyframe_loss": keyframe_loss,
            "train_loss": loss,
            "loss": loss,
            "keyframe_loc_metric_time_dist_train": keyframe_avg_time_dist
        }


class StateChangeClassification(VideoTask):
    checkpoint_metric = "state_change_metric_val"

    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        state_change_preds = self.forward(frames)

        state_change_loss = torch.mean(self.loss_fun(
            state_change_preds.squeeze(),
            state_change_label.long().squeeze()
        ))

        loss = state_change_loss

        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )

        return {
            "state_change_loss": state_change_loss,
            "train_loss": loss,
            "loss": loss,
            "state_change_metric_train": accuracy
        }

    def training_epoch_end(self, training_step_outputs):
        keys = [x for x in training_step_outputs[0].keys()]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].mean() for item in training_step_outputs]
                )
            ).mean()
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        state_change_preds = self.forward(frames)

        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )

        return {
            "state_change_metric_val": accuracy
        }

    def validation_epoch_end(self, validation_step_outputs):
        keys = [x for x in validation_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].mean() for item in validation_step_outputs]
                )
            ).mean()
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        state_change_preds = self.forward(frames)
        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )

        return {
            "labels": labels,
            "state_change_metric": accuracy
        }

    def test_epoch_end(self, test_step_outputs):
        keys = [x for x in test_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].mean() for item in test_step_outputs]
                )
            ).mean()
            self.log(key, metric, prog_bar=True)


class StateChangeAndKeyframeLocalisation(VideoTask):
    checkpoint_metric = "state_change_metric_val"

    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        # print('state change label', state_change_label)
        keyframe_preds, state_change_preds = self.forward(frames)

        pred = torch.sigmoid(keyframe_preds.squeeze(dim=1))
        keyframe_loss = bce(pred, labels.float()).sum(dim=1)

        # keyframe_preds_ = keyframe_preds.permute(0, 2, 1)  # (bs, 16, 1)
        #
        # keyframe_loss = self.loss_fun(
        #     keyframe_preds_.squeeze(dim=-1),  # (bs, 16)
        #     torch.argmax(labels.long(), dim=1)  # bs
        # )

        # We want to calculate the keyframe loss only for the clips with state
        # change

        # print(state_change_label.shape, keyframe_loss.shape)
        keyframe_loss = torch.mean(state_change_label.T * keyframe_loss)

        state_change_loss = torch.mean(self.loss_fun(
            state_change_preds.squeeze(),
            state_change_label.long().squeeze()
        ))

        lambda_1 = self.cfg.MODEL.LAMBDA_1
        lambda_2 = self.cfg.MODEL.LAMBDA_2
        loss = (keyframe_loss * lambda_2) + (state_change_loss * lambda_1)
        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_label,
            fps,
            info
        )

        return {
            "keyframe_loss": keyframe_loss,
            "state_change_loss": state_change_loss,
            "train_loss": loss,
            "loss": loss,
            "state_change_metric_train": accuracy,
            "keyframe_loc_metric_time_dist_train": keyframe_avg_time_dist
        }

    def training_epoch_end(self, training_step_outputs):
        keys = [x for x in training_step_outputs[0].keys()]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].mean() for item in training_step_outputs]
                )
            ).mean()
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        keyframe_preds, state_change_preds = self.forward(frames)

        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_label,
            fps,
            info,
            sum=True
        )
        state_change_count = state_change_label.sum()

        return {
            "state_change_metric_val": accuracy,
            "keyframe_loc_metric_time_dist_val": keyframe_avg_time_dist,
            "state_change_count": state_change_count
        }

    # def validation_epoch_end(self, validation_step_outputs):
    #     keys = [x for x in validation_step_outputs[0].keys() if "metric" in x]
    #     for key in keys:
    #         metric = torch.Tensor.float(
    #             torch.Tensor(
    #                 [item[key].mean() for item in validation_step_outputs]
    #             )
    #         ).mean()
    #         self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def validation_epoch_end(self, validation_step_outputs):
        # Fetch all outputs from all processes
        all_outs = self.all_gather(validation_step_outputs)

        state_change_count = sum([i['state_change_count'].sum() for i in all_outs])
        keyframe_distance_sum = sum([i['keyframe_loc_metric_time_dist_val'].sum() for i in all_outs])
        keyframe_loc_metric = keyframe_distance_sum / state_change_count

        state_change_acc = torch.FloatTensor([i['state_change_metric_val'].mean() for i in all_outs]).mean()

        # Log/write files on the rank zero process only
        if self.trainer.is_global_zero:
            self.log("state_change_metric_val", state_change_acc, rank_zero_only=True)
            self.log("keyframe_loc_metric_time_dist_val", keyframe_loc_metric, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        keyframe_preds, state_change_preds = self.forward(frames)
        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_label,
            fps,
            info
        )

        return {
            "labels": labels,
            "preds": keyframe_preds,
            "state_change_metric": accuracy,
            "keyframe_loc_metric_time": keyframe_avg_time_dist
        }

    def test_epoch_end(self, test_step_outputs):
        keys = [x for x in test_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].mean() for item in test_step_outputs]
                )
            ).mean()
            self.log(key, metric, prog_bar=True)
