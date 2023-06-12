#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import copy
import math
import torch
import torch.nn as nn
from models.pnr.video_model_builder import KeyframeLocalizationResNet, StateChangeClsResNet
from utils.pnr.parser import load_config_file
from utils.multitask.load_model import load_checkpoint, freeze_params


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


class TaskPromptTransformer2Task(nn.Module):
    def __init__(self, args, vocab, oscc_no_temp_pool=True):
        super(TaskPromptTransformer2Task, self).__init__()
        self.args = args
        self.vocab = vocab
        self.dim = args.hidden_dim
        self.n_tasks = 2
        self.task_dict = {'pnr': 0, 'oscc': 1}

        # fixed for now
        self.n_heads = args.num_heads
        self.num_layers = args.num_layers
        self.dp_rate = args.dropout

        # transformer encoder and decoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.n_heads, dropout=self.dp_rate),
            num_layers=self.num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=CustomDecoderLayer(d_model=self.dim, nhead=self.n_heads, dropout=self.dp_rate),
            num_layers=self.num_layers
        )
        self.proj_pnr = nn.Linear(8192, self.dim)
        self.proj_oscc = nn.Linear(8192, self.dim)

        self.fc = nn.Linear(self.dim, len(self.vocab))
        self.ln = nn.LayerNorm(self.dim)
        self.task_embed = nn.Parameter(torch.randn(1, self.n_tasks, self.dim), requires_grad=True)
        self.pos_embed = PositionalEncoding(self.dim, dropout=0.1)
        self.embedding = nn.Embedding(len(self.vocab), self.dim)

        # max sequence length ??
        self.seq_len = 5
        self.y_mask = self.get_tgt_mask(self.seq_len)

        self._init_parameters()

        # load task-specific models
        cfg_pnr = load_config_file(args.pnr_cfg_file)
        self.pnr_model = KeyframeLocalizationResNet(cfg_pnr)
        load_checkpoint(self.pnr_model, cfg_pnr.MISC.CHECKPOINT_FILE_PATH)
        freeze_params(self.pnr_model)

        cfg_oscc = load_config_file(args.oscc_cfg_file)
        cfg_oscc.MODEL.NO_TEMP_POOL = oscc_no_temp_pool     #args.oscc_no_temp_pool
        self.oscc_model = StateChangeClsResNet(cfg_oscc)
        load_checkpoint(self.oscc_model, cfg_oscc.MISC.CHECKPOINT_FILE_PATH)
        freeze_params(self.oscc_model)

        self.cfg_pnr = cfg_pnr
        self.cfg_oscc = cfg_oscc

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        return mask

    def encode_prepare(self, x, task_id):
        x = self.ln(x) + self.task_embed[:, task_id, :]
        x = x.permute(1, 0, 2)  #(#frame, bs, 256)
        x = self.pos_embed(x)
        return x


class TaskTranslationPromptTransformer2Task(TaskPromptTransformer2Task):
    def __init__(self, args, vocab):
        super(TaskTranslationPromptTransformer2Task, self).__init__(args, vocab)

    def encode(self, video_pnr):
        video_oscc = video_pnr.copy()
        with torch.no_grad():
            feat_pnr = self.pnr_model(video_pnr, middle=True)
            feat_oscc = self.oscc_model(video_oscc, middle=True)

        feat1 = self.proj_pnr(feat_pnr)
        feat2 = self.proj_oscc(feat_oscc)
        x1 = self.encode_prepare(feat1, 0)
        x2 = self.encode_prepare(feat2, 1)
        x = torch.cat((x1, x2), dim=0)  # (48, bs, 256)
        encoded_x = self.transformer_encoder(x)
        return encoded_x

    def decode(self, y, encoded_x):
        sy = y.size(1)
        y = y.permute(1, 0)  #(bs, seq_y)->(seq_y, bs)
        y = self.embedding(y) * math.sqrt(self.dim)  #(seq_y, bs, dim)
        y = self.pos_embed(y)
        # y_mask = self.get_tgt_mask(sequence_length).type_as(encoded_x)
        y_mask = self.y_mask[:sy, :sy].type_as(encoded_x)
        output = self.transformer_decoder(y, encoded_x, y_mask)
        output = self.fc(output)  #(seq_y, bs, vocab_size)
        return output

    def forward(self, video_pnr, target):
        encoded_x = self.encode(video_pnr)
        output = self.decode(target, encoded_x)
        return output.permute(1, 2, 0)

    def predict(self, video_pnr, task):
        assert task in ['pnr', 'oscc']
        batch_size = video_pnr[0].shape[0]
        encoded_x = self.encode(video_pnr)
        y = torch.ones((batch_size, 1)) * self.vocab[task]
        y = y.type_as(video_pnr[0]).long()
        output = self.decode(y, encoded_x)
        return output[0, :]


