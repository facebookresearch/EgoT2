#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import inspect
import random
import heapq
from torch.nn.init import xavier_uniform_
import collections
import torch
from torch.distributions.categorical import Categorical
from functools import reduce
import torch.nn.functional as F
import math
import copy
from einops import rearrange
import torch.nn as nn

from functools import reduce
from operator import mul
from .head_helper import MultiTaskHead, MultiTaskMViTHead
from .video_model_builder import SlowFast, _POOL1, MViT
from .build import MODEL_REGISTRY

from utils.multitask.build_vocab import vocab_idx_to_orig


class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CustomDecoderLayer, self).__init__(d_model, nhead, dropout=dropout)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)[0]
        return self.dropout2(x)


class PositionalEncoding(nn.Module):
    """
    Classic Attention-is-all-you-need positional encoding.
    From PyTorch docs.
    """

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)   #(max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


@MODEL_REGISTRY.register()
class ForecastingEncoderSeqDecoder(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.v_idx, self.n_idx = vocab_idx_to_orig()
        self.build_clip_backbone()
        self.build_encoder_decoder()

    # to encode frames into a set of {cfg.FORECASTING.NUM_INPUT_CLIPS} clips
    def build_clip_backbone(self):
        cfg = self.cfg
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        backbone_config = copy.deepcopy(cfg)
        backbone_config.MODEL.NUM_CLASSES = [cfg.MODEL.MULTI_INPUT_FEATURES]
        backbone_config.MODEL.HEAD_ACT = None


        if cfg.MODEL.ARCH == "mvit":
            self.backbone = MViT(backbone_config, with_head=True)
        else:
            self.backbone = SlowFast(backbone_config, with_head=True)
        # replace with:
        # self.backbone = MODEL_REGISTRY.get(cfg.FORECASTING.BACKBONE)(backbone_config, with_head=True)

        if cfg.MODEL.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Never freeze head.
            for param in self.backbone.head.parameters():
                param.requires_grad = True


    def build_encoder_decoder(self):
        # transformer encoder and decoder
        self.n_heads = self.cfg.MODEL.TRANSFORMER_ENCODER_HEADS    # 8
        self.n_layers = self.cfg.MODEL.TRANSFORMER_ENCODER_LAYERS   # 6
        self.dim = self.cfg.MODEL.MULTI_INPUT_FEATURES    # 2048
        self.ln = nn.LayerNorm(self.dim)
        self.pos_embed = PositionalEncoding(self.dim, dropout=0.1)
        self.embedding = nn.Embedding(len(self.vocab), self.dim)
        self.y_mask = self.get_tgt_mask(200)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.n_heads),
            num_layers=self.n_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=CustomDecoderLayer(d_model=self.dim, nhead=self.n_heads),
            num_layers=self.n_layers
        )
        self.fc = nn.Linear(self.dim, len(self.vocab))

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    # input = [(B, num_inp, 3, T, H, W), (B, num_inp, 3, T', H, W)]
    def encode_clips(self, x):
        # x -> [torch.Size([2, 2, 3, 8, 224, 224]), torch.Size([2, 2, 3, 32, 224, 224])]
        assert isinstance(x, list) and len(x) >= 1

        num_inputs = x[0].shape[1]
        batch = x[0].shape[0]
        features = []
        for i in range(num_inputs):
            pathway_for_input = []
            for pathway in x:
                input_clip = pathway[:, i]
                pathway_for_input.append(input_clip)

            # pathway_for_input -> [torch.Size([2, 3, 8, 224, 224]), torch.Size([2, 3,32, 224, 224])]
            input_feature = self.backbone(pathway_for_input)
            features.append(input_feature)

        return features

    def encode(self, x):
        x = torch.stack(x, dim=1).transpose(0, 1)  # (num_inputs, batch_size, dim)
        x = self.ln(x)
        x = self.pos_embed(x)
        x = self.transformer_encoder(x)
        return x  # (num_inputs, batch_size, dim) or x[-1]

    def decode(self, y, encoded_x):
        sy = y.size(1)
        y = y.permute(1, 0)  # (bs, seq_y)->(seq_y, bs)
        y = self.embedding(y) * math.sqrt(self.dim)  # (seq_y, bs, dim)
        y = self.pos_embed(y)
        # y_mask = self.get_tgt_mask(sequence_length).type_as(encoded_x)
        y_mask = self.y_mask[:sy, :sy].type_as(encoded_x)
        output = self.transformer_decoder(y, encoded_x, y_mask)
        output = self.fc(output)  # (seq_y, bs, vocab_size)
        return output

    def forward(self, x, target):
        features = self.encode_clips(x)  # [(bs, 4096)] list len=num_clips
        encoded_x = self.encode(features)
        output = self.decode(target, encoded_x)
        return output.permute(1, 2, 0)   # (bs, vocab_size, seq_y)

    def predict(self, x):
        features = self.encode_clips(x)  # [(bs, 4096)] list len=num_clips
        encoded_x = self.encode(features)
        batch_size = encoded_x.shape[1]

        seq_len = 41
        output_tokens = (torch.ones((batch_size, seq_len))).type_as(x[0]).long()
        output_tokens[:, 0] = self.vocab['action']  #todo: change action to lta
        preds_verb_list, preds_noun_list = [], []
        for sy in range(1, seq_len):
            y = output_tokens[:, :sy]
            output_prob = self.decode(y, encoded_x)
            if sy % 2 == 1:
                preds_verb_list.append(output_prob[-1, :, self.v_idx])    # map to verb idx
            else:
                preds_noun_list.append(output_prob[-1, :, self.n_idx])    # map to noun idx
            output = torch.argmax(output_prob, dim=-1)    #todo: better decoding ways
            output_tokens[:, sy] = output[-1, :]
        preds_verb = torch.stack(preds_verb_list, dim=1)
        preds_noun = torch.stack(preds_noun_list, dim=1)
        return [preds_verb, preds_noun]


    def generate(self, x, k=1):
        x = self.predict(x)
        results = []
        for head_x in x:
            if k>1:
                preds_dist = Categorical(logits=head_x)
                preds = [preds_dist.sample() for _ in range(k)]
            elif k==1:
                preds = [head_x.argmax(2)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)

        return results


@MODEL_REGISTRY.register()
class ForecastingEncoderSeparateSeqDecoder(ForecastingEncoderSeqDecoder):
    def __init__(self, cfg, vocab):
        super(ForecastingEncoderSeparateSeqDecoder, self).__init__(cfg, vocab)

    def predict(self, x):
        with torch.no_grad():
            features = self.encode_clips(x)  # [(bs, 4096)] list len=num_clips
            encoded_x = self.encode(features)

            batch_size = encoded_x.shape[1]
            y_verb = torch.ones((batch_size, 1)) * self.vocab['lta_verb']
            y_verb = y_verb.type_as(x[0]).long()
            output_verb = self.decode(y_verb, encoded_x)
            preds_verb = output_verb[0, :, self.v_idx].unsqueeze(dim=1)

            y_noun = torch.ones((batch_size, 1)) * self.vocab['lta_noun']
            y_noun = y_noun.type_as(x[0]).long()
            output_noun = self.decode(y_noun, encoded_x)
            preds_noun = output_noun[0, :, self.n_idx].unsqueeze(dim=1)

        return [preds_verb, preds_noun]
