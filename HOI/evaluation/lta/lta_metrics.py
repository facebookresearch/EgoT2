#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

"""Functions for computing metrics."""

import numpy as np
import torch

import editdistance

from utils.lta import distributed as du
from utils.lta import logging

from sklearn.metrics import average_precision_score

logger = logging.get_logger(__name__)


def distributed_topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k. Average reduces the result with all other
    distributed processes.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
    labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
    errors = topk_errors(preds, labels, ks)
    return errors


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"

    # Find the top max_k predictions for each sample
    maxk = max(ks)
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, maxk, dim=1, largest=True, sorted=True
    )

    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]

def edit_distance(preds, labels):
    """
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
    Lowest among K predictions
    """
    N, Z, K = preds.shape
    dists = []
    for n in range(N):
        dist = min([editdistance.eval(preds[n, :, k], labels[n])/Z for k in range(K)])
        dists.append(dist)
    return np.mean(dists)

def distributed_edit_distance(preds, labels):
    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
    labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
    return edit_distance(preds, labels)

def AUED(preds, labels):
    N, Z, K = preds.shape
    preds = preds.numpy()  # (N, Z, K)
    labels = labels.squeeze(-1).numpy()  # (N, Z)
    ED = np.vstack(
        [edit_distance(preds[:, :z], labels[:, :z]) for z in range(1, Z + 1)]
    )
    AUED = np.trapz(y=ED, axis=0) / (Z - 1)

    output = {"AUED": AUED}
    output.update({f"ED_{z}": ED[z] for z in range(Z)})
    return output

def distributed_AUED(preds, labels):
    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
    labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
    return AUED(preds, labels)


from utils.multitask.build_vocab import map_label_to_action

class ActionMetric:
    def __init__(self, vocab):
        super(ActionMetric).__init__()
        self.vocab = vocab
        verb_dict, noun_dict = map_label_to_action()
        self.verb_mapping = self._map_vocab_to_orig_idx(verb_dict)
        self.noun_mapping = self._map_vocab_to_orig_idx(noun_dict)

    def _map_vocab_to_orig_idx(self, dict):
        vocab2orig = {}
        for orig_idx, action in dict.items():
            vocab_idx = self.vocab[action]
            vocab2orig[vocab_idx] = orig_idx
        return vocab2orig

    def vocab_to_orig_idx(self, pred, mapping):
        result = []
        for v in pred:
            v = v.item()
            orig_idx = -1 if v not in mapping else mapping[v]
            result.append(orig_idx)
        return np.array(result)

    def top1_error(self, preds, labels):
        pred_verb = self.vocab_to_orig_idx(preds[:, 0], self.verb_mapping)
        labels_verb = labels[:, 0]
        verb_err = (pred_verb == -1).sum() / len(labels_verb)
        pred_verb = torch.Tensor(pred_verb).type_as(labels_verb)
        verb_acc = (pred_verb == labels_verb).sum() / len(labels_verb)

        pred_noun = self.vocab_to_orig_idx(preds[:, 1], self.noun_mapping)
        labels_noun = labels[:, 1]
        noun_err = (pred_noun == -1).sum() / len(labels_noun)
        pred_noun = torch.Tensor(pred_noun).type_as(labels_noun)
        noun_acc = (pred_noun == labels_noun).sum() / len(labels_noun)

        return verb_acc.item(), noun_acc.item(), verb_err, noun_err


from torchmetrics import Metric
class ARMetric(Metric):
    def __init__(self, vocab):
        super().__init__()
        self.add_state("v_cnt", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("v_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("v_err", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_cnt", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_err", default=torch.tensor(0), dist_reduce_fx="sum")
        self.vocab = vocab
        verb_dict, noun_dict = map_label_to_action()
        self.verb_mapping = self._map_vocab_to_orig_idx(verb_dict)
        self.noun_mapping = self._map_vocab_to_orig_idx(noun_dict)

    def _map_vocab_to_orig_idx(self, dict):
        vocab2orig = {}
        for orig_idx, action in dict.items():
            vocab_idx = self.vocab[action]
            vocab2orig[vocab_idx] = orig_idx
        return vocab2orig

    def vocab_to_orig_idx(self, pred, mapping):
        result = []
        for v in pred:
            v = v.item()
            orig_idx = -1 if v not in mapping else mapping[v]
            result.append(orig_idx)
        return np.array(result)

    def update(self, preds, labels):
        pred_verb = self.vocab_to_orig_idx(preds[:, 0], self.verb_mapping)
        labels_verb = labels[:, 0]
        self.v_cnt += len(labels_verb)
        self.v_err += (pred_verb == -1).sum()
        pred_verb = torch.Tensor(pred_verb).type_as(labels_verb)
        self.v_correct += (pred_verb == labels_verb).sum()

        pred_noun = self.vocab_to_orig_idx(preds[:, 1], self.noun_mapping)
        labels_noun = labels[:, 1]
        self.n_cnt +=  len(labels_noun)
        self.n_err += (pred_noun == -1).sum()
        pred_noun = torch.Tensor(pred_noun).type_as(labels_noun)
        self.n_correct += (pred_noun == labels_noun).sum()


    def compute(self):
        return self.v_err.float() / self.v_cnt, self.n_err.float() / self.n_cnt, \
               self.v_correct.float() / self.v_cnt, self.n_correct.float() / self.n_cnt



class LTAMetricSimple(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("v_correct", default=[], dist_reduce_fx="cat")
        self.add_state("v_error", default=[], dist_reduce_fx="cat")
        self.add_state("n_correct", default=[], dist_reduce_fx="cat")
        self.add_state("n_error", default=[], dist_reduce_fx="cat")

    def update(self, preds, labels):
        preds_verb = preds[:, 0]
        preds_noun = preds[:, 1]
        print('xxx')


class LTAMetric(Metric):
    def __init__(self, vocab):
        super().__init__()
        # self.add_state("v_cnt", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("v_correct", default=[], dist_reduce_fx="cat")
        self.add_state("v_error", default=[], dist_reduce_fx="cat")
        # self.add_state("n_cnt", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_correct", default=[], dist_reduce_fx="cat")
        self.add_state("n_error", default=[], dist_reduce_fx="cat")
        self.add_state("unique_id_list", default=[], dist_reduce_fx="cat")
        self.vocab = vocab
        verb_dict, noun_dict = map_label_to_action()
        self.verb_mapping = self._map_vocab_to_orig_idx(verb_dict)
        self.noun_mapping = self._map_vocab_to_orig_idx(noun_dict)

    def _construct_unique_id_mapping(self, clip_annotations):
        self.lta_mapping = dict()
        for i, annotations in enumerate(clip_annotations):
            clips = annotations[1]['input_clips'][-1]
            clip_name = clips["clip_uid"] + "_" + str(clips["action_idx"])
            if clip_name in self.lta_mapping:
                print('duplicate', clip_name)
            self.lta_mapping[clip_name] = i

        print('LTA Unique id map dict len', len(self.lta_mapping))

    def _map_vocab_to_orig_idx(self, dict):
        vocab2orig = {}
        for orig_idx, action in dict.items():
            vocab_idx = self.vocab[action]
            vocab2orig[vocab_idx] = orig_idx
        return vocab2orig

    def vocab_to_orig_idx(self, pred, mapping):
        result = []
        for v in pred:
            v = v.item()
            orig_idx = -1 if v not in mapping else mapping[v]
            result.append(orig_idx)
        return np.array(result)

    def update(self, preds, labels, unique_ids):
        preds_verb = self.vocab_to_orig_idx(preds[:, 0], self.verb_mapping)
        preds_noun = self.vocab_to_orig_idx(preds[:, 1], self.noun_mapping)
        for pred_v, label_v, pred_n, label_n, unique_id in zip(preds_verb, labels[:, 0], preds_noun, labels[:, 1], unique_ids):
            uid = torch.Tensor([self.lta_mapping[unique_id]]).to(label_v.device)
            self.unique_id_list.append(uid)
            if pred_v == -1:
                self.v_error.append(torch.ones_like(uid))
            else:
                self.v_error.append(torch.zeros_like(uid))
            if pred_v == label_v.item():
                self.v_correct.append(torch.ones_like(uid))
            else:
                self.v_correct.append(torch.zeros_like(uid))

            if pred_n == -1:
                self.n_error.append(torch.ones_like(uid))
            else:
                self.n_error.append(torch.zeros_like(uid))
            if pred_n == label_n.item():
                self.n_correct.append(torch.ones_like(uid))
            else:
                self.n_correct.append(torch.zeros_like(uid))

    def compute(self):
        v_error, v_correct, cnt = 0.0, 0.0, 0.0
        n_error, n_correct = 0.0, 0.0
        tmp_list = []
        for i, uid in enumerate(self.unique_id_list):
            id = uid.item()
            if id in tmp_list:
                print(id, 'in tmp list')
                # continue
            tmp_list.append(id)
            cnt = cnt + 1
            v_error = v_error + self.v_error[i].item()
            v_correct = v_correct + self.v_correct[i].item()
            n_error = n_error + self.n_error[i].item()
            n_correct = n_correct + self.n_correct[i].item()

        return v_error / cnt, v_correct / cnt, n_error / cnt, n_correct / cnt, cnt
