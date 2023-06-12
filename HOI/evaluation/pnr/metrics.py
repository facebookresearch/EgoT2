#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def state_change_accuracy(preds, labels):
    correct = 0
    total = 0
    for pred, label in zip(preds, labels):
        pred_ = torch.argmax(pred)
        if pred_.item() == label.item():
            correct += 1
        total += 1
    accuracy = correct/total
    return accuracy


def keyframe_accuracy(preds, labels, sc_labels):
    correct, total = 0, 0
    for pred, label, sc_label in zip(preds, labels, sc_labels):
        pred_ = torch.argmax(pred)
        label_ = torch.argmax(label)
        if sc_label.item() == 1:
            total += 1
            if pred_.item() == label_.item():
                correct += 1
    # print(correct, total)
    return correct, total


def keyframe_distance(
    preds,
    labels,
    sc_labels,
    fps,
    info,
    evaluate_trained=False,
    sum=False
):
    distance_list = list()
    for pred, label, sc_label, ind_fps, start_frame, end_frame, pnr_frame in zip(
        preds,
        labels,
        sc_labels,
        fps,
        info['clip_start_frame'],
        info['clip_end_frame'],
        info['pnr_frame']
    ):
        if sc_label.item() == 1:
            keyframe_loc_pred = torch.argmax(pred).item()
            keyframe_loc_pred_mapped = (
                end_frame - start_frame
            ) / 16 * keyframe_loc_pred
            keyframe_loc_pred_mapped = keyframe_loc_pred_mapped.item()
            gt = pnr_frame.item() - start_frame.item()

            # naive_pred = 0.5 * (end_frame - start_frame)
            # print('gt | pred | naive pred', gt, keyframe_loc_pred_mapped, naive_pred.item())
            # err_frame = abs(naive_pred - gt)
            err_frame = abs(keyframe_loc_pred_mapped - gt)
            err_sec = err_frame/ind_fps.item()
            distance_list.append(err_sec)
    if len(distance_list) == 0:
        # If evaluating the trained model, use this
        if evaluate_trained:
            return None
        # Otherwise, Lightning expects us to give a number.
        # Due to this, the Tensorboard graphs' results for keyframe distance
        # will be a little inaccurate.
        return np.mean(0.0)
    if sum:
        return np.sum(distance_list)
    else:
        return np.mean(distance_list)


class PnrOsccMetric:
    def __init__(self, vocab):
        super(PnrOsccMetric).__init__()
        self.vocab = vocab
        self.pnr_indices = self._get_pnr_idx_invocab()
        self.oscc_indices = self._get_oscc_idx_invocab()
        print('pnr', self.pnr_indices)
        print('oscc', self.oscc_indices)

    def _get_pnr_idx_invocab(self):
        result = []
        for i in range(16):
            result.append(self.vocab[str(i)])
        return result

    def _get_oscc_idx_invocab(self):
        result = [self.vocab['False'], self.vocab['True']]
        return result

    def pnr_distance(self, preds, labels, fps, info):
        distance_list = list()
        err = 0
        for pred, label, ind_fps, start_frame, end_frame, pnr_frame in zip(preds, labels, fps,
            info['clip_start_frame'], info['clip_end_frame'], info['pnr_frame']):
            pred_idx = torch.argmax(pred).item()
            if pred_idx not in self.pnr_indices:
                err = err + 1
            pred_subset = pred[self.pnr_indices]
            pred_idx = torch.argmax(pred_subset).item()
            keyframe_loc_pred_mapped = (end_frame - start_frame) / 16 * pred_idx
            keyframe_loc_pred_mapped = keyframe_loc_pred_mapped.item()
            gt = pnr_frame.item() - start_frame.item()
            err_frame = abs(keyframe_loc_pred_mapped - gt)
            err_sec = err_frame / ind_fps.item()
            distance_list.append(err_sec)

        return np.mean(distance_list), err/len(distance_list)

    def oscc_acc(self, preds, labels):
        correct = 0
        error = 0
        total = 0
        for pred, label in zip(preds, labels):
            pred_idx = torch.argmax(pred).item()
            if pred_idx not in self.oscc_indices:
                error += 1
            pred_subset = pred[self.oscc_indices]
            pred_idx = torch.argmax(pred_subset).item()
            if pred_idx == label.item():
                correct += 1
            total += 1
        return correct / total, error / total



from torchmetrics import Metric
class PNRMetric(Metric):
    def __init__(self, vocab):
        super().__init__()
        # self.add_state("cnt", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.add_state("err", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.add_state("dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # self.add_state("cnt", default=[], dist_reduce_fx="cat")
        self.add_state("err", default=[], dist_reduce_fx="cat")
        self.add_state("dist", default=[], dist_reduce_fx="cat")
        self.add_state("unique_id_list", default=[], dist_reduce_fx="cat")
        self.vocab = vocab
        self.pnr_indices = self._get_pnr_idx_invocab()


    def _construct_unique_id_mapping(self, package):
        self.pnr_dict = dict()
        for key, item in package.items():
            if item['unique_id'] in self.pnr_dict:
                print('duplicate', item)
            self.pnr_dict[item['unique_id']] = key
        check_list = [key for key, item in self.pnr_dict.items()]
        check_list = set(check_list)
        print('PNR Unique id map dict len', len(self.pnr_dict), len(check_list))

    def _get_pnr_idx_invocab(self):
        result = []
        for i in range(16):
            result.append(self.vocab[str(i)])
        return result

    def update(self, preds, labels, fps, info):
        for pred, label, ind_fps, start_frame, end_frame, pnr_frame, unique_id in \
                zip(preds, labels, fps, info['clip_start_frame'], info['clip_end_frame'], info['pnr_frame'], info['unique_id']):
            uid = torch.Tensor([self.pnr_dict[unique_id]]).to(pred.device)
            self.unique_id_list.append(uid)
            pred_idx = torch.argmax(pred).item()
            if pred_idx not in self.pnr_indices:
                self.err.append(torch.ones_like(uid))
            else:
                self.err.append(torch.zeros_like(uid))
            pred_subset = pred[self.pnr_indices]
            pred_idx = torch.argmax(pred_subset).item()
            keyframe_loc_pred_mapped = (end_frame - start_frame) / 16 * pred_idx
            keyframe_loc_pred_mapped = keyframe_loc_pred_mapped.item()
            gt = pnr_frame.item() - start_frame.item()
            err_frame = abs(keyframe_loc_pred_mapped - gt)
            err_sec = err_frame / ind_fps.item()
            self.dist.append(torch.Tensor([err_sec]).to(pred.device))


    def compute(self):
        err, cnt, dist = 0.0, 0.0, 0.0
        tmp_list = []
        for i, uid in enumerate(self.unique_id_list):
            id = uid.item()
            if id in tmp_list:
                print(id, 'in tmp list')
                # continue
            tmp_list.append(id)
            cnt = cnt + 1
            err = err + self.err[i].item()
            dist = dist + self.dist[i].item()
        return err / cnt, dist / cnt, cnt


class OSCCMetric(Metric):
    def __init__(self, vocab):
        super().__init__()
        # self.add_state("cnt", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.add_state("error", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("error", default=[], dist_reduce_fx="cat")
        self.add_state("correct", default=[], dist_reduce_fx="cat")
        self.add_state("unique_id_list", default=[], dist_reduce_fx="cat")
        self.vocab = vocab
        self.oscc_indices = self._get_oscc_idx_invocab()

    def _construct_unique_id_mapping(self, package):
        self.oscc_dict = dict()
        for key, item in package.items():
            if item['unique_id'] in self.oscc_dict:
                print('duplicate', item)
            self.oscc_dict[item['unique_id']] = key
        print('OSCC Unique id map dict len', len(self.oscc_dict))

    def _get_oscc_idx_invocab(self):
        result = [self.vocab['False'], self.vocab['True']]
        return result

    def update(self, preds, labels, info):
        for pred, label, unique_id in zip(preds, labels, info['unique_id']):
            uid = torch.Tensor([self.oscc_dict[unique_id]]).to(pred.device)
            self.unique_id_list.append(uid)
            pred_idx = torch.argmax(pred).item()
            if pred_idx not in self.oscc_indices:
                self.error.append(torch.ones_like(uid))
            else:
                self.error.append(torch.zeros_like(uid))
            pred_subset = pred[self.oscc_indices]
            pred_idx = torch.argmax(pred_subset).item()
            if pred_idx == label.item():
                self.correct.append(torch.ones_like(uid))
            else:
                self.correct.append(torch.zeros_like(uid))


    def compute(self):
        error, correct, cnt = 0.0, 0.0, 0.0
        tmp_list = []
        for i, uid in enumerate(self.unique_id_list):
            id = uid.item()
            if id in tmp_list:
                print(id, 'in tmp list')
                # continue
            tmp_list.append(id)
            cnt = cnt + 1
            error = error + self.error[i].item()
            correct = correct + self.correct[i].item()
        return error / cnt, correct / cnt, cnt
