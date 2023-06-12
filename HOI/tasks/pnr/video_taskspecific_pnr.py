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


class KeyframeLocalisation2Loader(VideoTask):
    checkpoint_metric = "keyframe_loc_metric_time_dist_val_neg"

    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch['orig']
        frames_lta = batch['recognition']
        keyframe_preds = self.model(frames, frames_lta)

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
        frames, labels, state_change_label, fps, info = batch['orig']
        frames_lta = batch['recognition']
        keyframe_preds = self.model(frames, frames_lta)
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
        frames, labels, state_change_label, fps, info = batch['orig']
        frames_lta = batch['recognition']
        keyframe_preds = self.model(frames, frames_lta)

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


class StateChangeClassification2Loader(VideoTask):
    checkpoint_metric = "state_change_metric_val"

    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch['orig']
        frames_lta = batch['recognition']
        state_change_preds = self.model(frames, frames_lta)

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
        frames, labels, state_change_label, fps, info = batch['orig']
        frames_lta = batch['recognition']
        state_change_preds = self.model(frames, frames_lta)

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
        frames, labels, state_change_label, fps, info = batch['orig']
        frames_lta = batch['recognition']
        state_change_preds = self.model(frames, frames_lta)

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
