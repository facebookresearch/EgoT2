#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torchtext.vocab import build_vocab_from_iterator


def map_label_to_action():
    data = json.load(open('/datasets01/ego4d_track2/v1/annotations/fho_lta_taxonomy.json', 'r'))
    v_list = []
    verb_dict = {}
    for i, verb in enumerate(data["verbs"]):
        v = verb.split('(')[0].replace('_', '')  # default one word for now, todo: allow 2+ words for verb and noun
        verb_dict[i] = v
        # if v in v_list:
        #     print(i, verb, v, 'already in list')
        v_list.append(v)

    n_list = []
    noun_dict = {}
    for i, noun in enumerate(data["nouns"]):
        n = noun.split('(')[0].replace('_', '')
        noun_dict[i] = n
        # if n in n_list:
        #     print(i, noun, n, 'already in list')
        n_list.append(n)

    noun_dict[19] = "bat_sports"
    noun_dict[20] = "bat_tool"
    noun_dict[84] = "chip_food"
    noun_dict[85] = "chip_wood\'"
    noun_dict[86] = "chip_wood"
    noun_dict[270] = "nut_food"
    noun_dict[271] = "nut_tool"
    noun_dict[320] = "pot_planter"

    # for i, verb in verb_dict.items():
    #     print(i, verb)
    # print('-' * 100)
    # for i, noun in noun_dict.items():
    #     print(i, noun)

    return verb_dict, noun_dict


def build_vocab():
    # tokens = ['pnr', 'oscc',' action', 'True', 'False', '</s>', '<unk>']
    tokens = ['pnr', 'oscc', 'action_verb', 'action_noun', 'lta_verb', 'lta_noun', 'True', 'False', '</s>', '<unk>']
    for i in range(16):
        tokens.append(str(i))
    verb_dict, noun_dict = map_label_to_action()
    for _, verb in verb_dict.items():
        tokens.append(verb)
    for _, noun in noun_dict.items():
        tokens.append(noun)
    # v = vocab(OrderedDict([(token, 1) for token in tokens]), specials=["</s>", "<unk>"])
    v = vocab(OrderedDict([(token, 1) for token in tokens]))
    v.set_default_index(v["<unk>"])
    print('vocab size', len(v))
    # print('vocab mapping', v.get_stoi())
    print(f"pnr {v['pnr']} | oscc {v['oscc']} | action verb {v['action_verb']} | action noun {v['action_noun']}")
    # ttm_idx = v['ttm']
    # ttm_str = v.get_itos()[ttm_idx]
    # print(ttm_idx, ttm_str)
    return v


def build_vocab_task12():
    tokens = ['pnr', 'oscc', 'True', 'False', '</s>', '<unk>']
    for i in range(16):
        tokens.append(str(i))
    v = vocab(OrderedDict([(token, 1) for token in tokens]))
    v.set_default_index(v["<unk>"])
    print('vocab size', len(v))
    return v


def build_vocab_task125():
    tokens = ['pnr', 'oscc', 'lam', 'True', 'False', '</s>', '<unk>']
    for i in range(16):
        tokens.append(str(i))
    v = vocab(OrderedDict([(token, 1) for token in tokens]))
    v.set_default_index(v["<unk>"])
    print('vocab size', len(v))
    return v


def build_vocab_orig():
    tokens = ['pnr', 'oscc',' action', 'True', 'False', '</s>', '<unk>']
    for i in range(16):
        tokens.append(str(i))
    verb_dict, noun_dict = map_label_to_action()
    for _, verb in verb_dict.items():
        tokens.append(verb)
    for _, noun in noun_dict.items():
        tokens.append(noun)
    v = vocab(OrderedDict([(token, 1) for token in tokens]), specials=["</s>", "<unk>"])
    # v = vocab(OrderedDict([(token, 1) for token in tokens]))
    v.set_default_index(v["<unk>"])
    print('vocab size', len(v))
    # print('vocab mapping', v.get_stoi())
    print(f"action {v['action']} | pnr {v['pnr']} | oscc {v['oscc']}")
    # ttm_idx = v['ttm']
    # ttm_str = v.get_itos()[ttm_idx]
    # print(ttm_idx, ttm_str)
    return v

def vocab_idx_to_orig():
    verb_dict, noun_dict = map_label_to_action()
    vocab = build_vocab()
    v_list = []
    n_list = []

    for i in range(len(verb_dict)):    # verb category
        verb_idx = vocab[verb_dict[i]]
        v_list.append(verb_idx)

    for i in range(len(noun_dict)):
        noun_idx = vocab[noun_dict[i]]
        n_list.append(noun_idx)

    return np.array(v_list), np.array(n_list)


# if __name__ == '__main__':
#     v_arr, n_arr = vocab_idx_to_orig()