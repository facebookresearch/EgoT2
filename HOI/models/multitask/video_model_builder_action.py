#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import math
import copy
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from models.lta.video_model_builder import SlowFast
from models.lta.lta_models import ForecastingEncoderDecoder
from models.lta.lta_models_seqdecoder import CustomDecoderLayer
from utils.lta.parser import load_config_from_file as load_lta_config
from utils.multitask.load_model import load_checkpoint, load_lta_backbone, freeze_backbone_params, freeze_params
from utils.multitask.build_vocab import vocab_idx_to_orig
from .video_model_builder import PositionalEncoding


class TaskTranslationPromptTransformerActionTask(nn.Module):
    def __init__(self, args, vocab):
        super(TaskTranslationPromptTransformerActionTask, self).__init__()
        self.args = args
        self.vocab = vocab
        self.v_idx, self.n_idx = vocab_idx_to_orig()
        self.dim = args.hidden_dim
        self.n_tasks = 2

        # fixed for now
        self.n_heads = args.num_heads
        self.dim_feedforward = args.ff_dim
        self.num_layers = args.num_layers
        self.dp_rate = args.dropout
        self.max_output_length = 500

        # transformer encoder and decoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.n_heads, dropout=self.dp_rate),
            num_layers=self.num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=CustomDecoderLayer(d_model=self.dim, nhead=self.n_heads, dropout=self.dp_rate),
            num_layers=self.num_layers
        )

        self.fc = nn.Linear(self.dim, len(self.vocab))
        self.ln = nn.LayerNorm(self.dim)
        self.task_embed = nn.Parameter(torch.randn(1, self.n_tasks, self.dim), requires_grad=True)
        self.pos_embed = PositionalEncoding(self.dim, dropout=self.dp_rate)
        self.pe = nn.Parameter(torch.randn(1, 4, self.dim), requires_grad=True)
        self.embedding = nn.Embedding(len(self.vocab), self.dim)

        # max sequence length ??
        self.seq_len = 200
        self.y_mask = self.get_tgt_mask(self.seq_len)

        self._init_parameters()

        # load task-specific models
        cfg = load_lta_config(args.lta_cfg_file)
        self.k = cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT
        backbone_cfg = copy.deepcopy(cfg)
        backbone_cfg.MODEL.NUM_CLASSES = [self.dim]
        backbone_cfg.MODEL.HEAD_ACT = None

        self.action_model = SlowFast(backbone_cfg, with_head=True)
        load_lta_backbone(self.action_model, cfg.CHECKPOINT_FILE_PATH_AR, True, True)
        freeze_backbone_params(self.action_model)

        lta_cfg = copy.deepcopy(cfg)
        lta_cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT = 20
        self.lta_model = ForecastingEncoderDecoder(lta_cfg)
        load_lta_backbone(self.lta_model, cfg.CHECKPOINT_FILE_PATH_LTA)
        freeze_params(self.lta_model)


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

    def encode_clips(self, model, x):
        # x -> [torch.Size([2, 2, 3, 8, 224, 224]), torch.Size([2, 2, 3, 32, 224, 224])]
        assert isinstance(x, list) and len(x) >= 1

        num_inputs = x[0].shape[1]
        features = []
        for i in range(num_inputs):
            pathway_for_input = []
            for pathway in x:
                input_clip = pathway[:, i]
                pathway_for_input.append(input_clip)

            # pathway_for_input -> [torch.Size([2, 3, 8, 224, 224]), torch.Size([2, 3,32, 224, 224])]
            input_feature = model(pathway_for_input)
            features.append(input_feature)

        return torch.stack(features, dim=1)  # (bs, num_inputs, d)

    def encode(self, video, task):
        if 'lta' in task:  # only use tokens produced by action models
            feat_action = self.encode_clips(self.action_model, video)  # (bs, num_input, d)
            feat_lta = self.lta_model(video, None, middle=True).transpose(0, 1)  # (bs, num_input, d)
            # x1 = self.encode_prepare(feat_action, 0)
            # x2 = self.encode_prepare(feat_lta, 1)
            # x = torch.cat((x1, x2), dim=0)  # (2*num_input, bs, d)
            feat = torch.cat((feat_action, feat_lta), dim=1)  # (bs, num_input*2, d)
            x = (self.ln(feat) + self.pe).transpose(0, 1)

        else:
            feat_action = self.action_model(video).unsqueeze(1)  # (bs, 1, d)
            x = self.encode_prepare(feat_action, 0)  # (1, bs, d)
        encoded_x = self.transformer_encoder(x)
        return encoded_x


    def decode(self, y, encoded_x):
        sy = y.size(1)
        y = y.permute(1, 0)  #(bs, seq_y)->(seq_y, bs)
        y = self.embedding(y) * math.sqrt(self.dim)  #(seq_y, bs, dim)
        y = self.pos_embed(y)
        y_mask = self.y_mask[:sy, :sy].type_as(encoded_x)
        output = self.transformer_decoder(y, encoded_x, y_mask)
        output = self.fc(output)  #(seq_y, bs, vocab_size)
        return output


    def forward(self, video, target, task):
        assert task in ['action_verb', 'action_noun', 'lta_verb', 'lta_noun']
        encoded_x = self.encode(video, task)
        # encoded_x = self.encode_temporal(video, task)
        output = self.decode(target, encoded_x) # (seq_y, bs, vocab_size)
        return output.permute(1, 2, 0)

    def predict(self, video, task):
        assert task in ['action', 'lta']
        encoded_x = self.encode(video, task)
        batch_size = encoded_x.shape[1]

        y_verb = torch.ones((batch_size, 1)) * self.vocab[task + '_verb']
        y_verb = y_verb.type_as(video[0]).long()
        output_verb = self.decode(y_verb, encoded_x)
        preds_verb = output_verb[0, :, self.v_idx]

        y_noun = torch.ones((batch_size, 1)) * self.vocab[task + '_noun']
        y_noun = y_noun.type_as(video[0]).long()
        output_noun = self.decode(y_noun, encoded_x)
        preds_noun = output_noun[0, :, self.n_idx]

        if task == 'lta':
            preds_verb = preds_verb.unsqueeze(dim=1)
            preds_noun = preds_noun.unsqueeze(dim=1)

        return [preds_verb, preds_noun]


    def generate(self, x):
        x = self.predict(x, 'lta')
        results = []
        for head_x in x:
            if self.k>1:
                preds_dist = Categorical(logits=head_x)
                preds = [preds_dist.sample() for _ in range(self.k)]
            elif self.k==1:
                preds = [head_x.argmax(2)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)

        return results


class TaskTranslationPromptTransformerTemporalActionTask(TaskTranslationPromptTransformerActionTask):
    def __init__(self, args, vocab):
        # args.hidden_dim = 256
        super(TaskTranslationPromptTransformerTemporalActionTask, self).__init__(args, vocab)
        self.proj_action_slow = nn.Linear(2048, self.dim)
        self.proj_action_fast = nn.Linear(256, self.dim)
        self.avg_pool_slow = nn.AdaptiveAvgPool3d((2, 1, 1))
        self.avg_pool_fast = nn.AdaptiveAvgPool3d((2, 1, 1))
        self.proj_lta = nn.Linear(2048, self.dim)

    def encode(self, video, task):
        if 'lta' in task:
            feat_action = self.encode_clips(self.action_model, video)  # (bs, num_input, dim)
            feat_lta = self.proj_lta(self.lta_model(video, None, middle=True).transpose(0, 1))  # (bs, num_input, dim)
            x1 = self.encode_prepare(feat_action, 0)
            x2 = self.encode_prepare(feat_lta, 1)
            x = self.ln(torch.cat((x1, x2), dim=0))  # (4, bs, dim), add layer norm

        else:
            feat_list = self.action_model(video, middle=True)
            feat1 = self.proj_action_slow(self.avg_pool_slow(feat_list[0]).squeeze(-1).squeeze(-1).permute(0, 2, 1))
            feat2 = self.proj_action_fast(self.avg_pool_fast(feat_list[1]).squeeze(-1).squeeze(-1).permute(0, 2, 1))
            feat = torch.cat((feat1, feat2), dim=1)  #(bs, 4, dim)
            x = self.encode_prepare(feat, 0)  #(4, bs, dim)
        encoded_x = self.transformer_encoder(x)
        return encoded_x

