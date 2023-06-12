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
from models.lta.video_model_builder import SlowFast
from models.lta.lta_models import ForecastingEncoderDecoder
from utils.pnr.parser import load_config_file
from utils.lta.parser import load_config_from_file as load_lta_config
# from utils.lta.parser import parse_args
from utils.multitask.load_model import load_checkpoint, load_lta_backbone, freeze_params, load_recognition_backbone, freeze_backbone_params


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


class TaskPromptTransformer(nn.Module):
    def __init__(self, args, vocab, oscc_no_temp_pool=True):
        super(TaskPromptTransformer, self).__init__()
        self.args = args
        self.vocab = vocab
        self.dim = args.hidden_dim
        self.n_tasks = 3
        self.task_dict = {'pnr': 0, 'oscc': 1, 'action': 2}

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
        self.proj_action_slow = nn.Linear(2048, self.dim)
        self.proj_action_fast = nn.Linear(256, self.dim)
        self.avg_pool_slow = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avg_pool_fast = nn.AdaptiveAvgPool3d((8, 1, 1))

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

        # args_tmp = parse_args()
        # args_tmp.cfg_file = args.action_cfg_file
        cfg_recognition = load_lta_config(args.action_cfg_file)
        cfg_recognition.MODEL.NUM_CLASSES = [self.dim]
        cfg_recognition.MODEL.HEAD_ACT = None
        self.recognition_model = SlowFast(cfg_recognition, with_head=True)
        load_recognition_backbone(self.recognition_model, cfg_recognition.CHECKPOINT_FILE_PATH)
        freeze_backbone_params(self.recognition_model)  # do not freeze head

        self.cfg_pnr = cfg_pnr
        self.cfg_oscc = cfg_oscc
        self.cfg_action = cfg_recognition

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

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def encode_prepare(self, x, task_id):
        x = self.ln(x) + self.task_embed[:, task_id, :]
        x = x.permute(1, 0, 2)  #(#frame, bs, 256)
        x = self.pos_embed(x)
        return x

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


    def forward(self, video, target, task):
        assert task in ['pnr', 'oscc', 'action']
        if task == 'pnr':
            with torch.no_grad():
                feat = self.pnr_model(video, middle=True)
            feat = self.proj_pnr(feat)
        elif task == 'oscc':
            with torch.no_grad():
                feat = self.oscc_model(video, middle=True)
            feat = self.proj_oscc(feat)
        elif task == 'action':
            with torch.no_grad():
                x_action_list = self.recognition_model(video, middle=True)
            feat1 = self.proj_action_slow(self.avg_pool_slow(x_action_list[0]).squeeze(-1).squeeze(-1).permute(0, 2, 1))
            feat2 = self.proj_action_fast(self.avg_pool_fast(x_action_list[1]).squeeze(-1).squeeze(-1).permute(0, 2, 1))
            feat = torch.cat((feat1, feat2), dim=1)

        x = self.encode_prepare(feat, self.task_dict[task])  # (16, bs, 256)
        encoded_x = self.transformer_encoder(x)  # discard this step?, same shape as x
        output = self.decode(target, encoded_x)
        return output.permute(1, 2, 0)


    def predict(self, video, task):
        assert task in ['pnr', 'oscc']
        batch_size = video[0].shape[0]
        with torch.no_grad():
            if task == 'pnr':
                feat = self.proj_pnr(self.pnr_model(video, middle=True))
            else:
                feat = self.proj_oscc(self.oscc_model(video, middle=True))
            x = self.encode_prepare(feat, self.task_dict[task])
            encoded_x = self.transformer_encoder(x)

        y = torch.ones((batch_size, 1)) * self.vocab[task]
        y = y.type_as(video[0]).long()
        output = self.decode(y, encoded_x)
        return output[0, :]

    def predict_ac(self, video):
        batch_size = video[0].shape[0]
        with torch.no_grad():
            x_action_list = self.recognition_model(video, middle=True)
            feat1 = self.proj_action_slow(self.avg_pool_slow(x_action_list[0]).squeeze(-1).squeeze(-1).permute(0, 2, 1))
            feat2 = self.proj_action_fast(self.avg_pool_fast(x_action_list[1]).squeeze(-1).squeeze(-1).permute(0, 2, 1))
            feat = torch.cat((feat1, feat2), dim=1)

        x = self.encode_prepare(feat, self.task_dict['action'])
        encoded_x = self.transformer_encoder(x)

        seq_len = 3  # set fixed output seq len for now
        output_tokens = (torch.ones((batch_size, seq_len))).type_as(video[0]).long()
        output_tokens[:, 0] = self.vocab['action']      # start token
        for sy in range(1, seq_len):
            y = output_tokens[:, :sy]
            output = self.decode(y, encoded_x)
            output = torch.argmax(output, dim=-1)
            output_tokens[:, sy] = output[-1, :]
        return output_tokens[:, 1:]  # idx in vocab


class TaskTranslationPromptTransformer(TaskPromptTransformer):
    def __init__(self, args, vocab):
        super(TaskTranslationPromptTransformer, self).__init__(args, vocab)

    def encode(self, video_pnr, video_ac):
        video_oscc = video_pnr.copy()
        with torch.no_grad():
            feat_pnr = self.pnr_model(video_pnr, middle=True)
            feat_oscc = self.oscc_model(video_oscc, middle=True)
            x_action_list = self.recognition_model(video_ac, middle=True)

        feat1 = self.proj_pnr(feat_pnr)
        feat2 = self.proj_oscc(feat_oscc)
        feat3_1 = self.proj_action_slow(self.avg_pool_slow(x_action_list[0]).squeeze(-1).squeeze(-1).permute(0, 2, 1))
        feat3_2 = self.proj_action_fast(self.avg_pool_fast(x_action_list[1]).squeeze(-1).squeeze(-1).permute(0, 2, 1))
        feat3 = torch.cat((feat3_1, feat3_2), dim=1)

        x1 = self.encode_prepare(feat1, 0)
        x2 = self.encode_prepare(feat2, 1)
        x3 = self.encode_prepare(feat3, 2)  # (16, bs, 256)
        x = torch.cat((x1, x2, x3), dim=0)  # (48, bs, 256)
        encoded_x = self.transformer_encoder(x)
        return encoded_x

    def forward(self, video_pnr, video_ac, target):
        encoded_x = self.encode(video_pnr, video_ac)
        output = self.decode(target, encoded_x) # todo: crop certain region of encoded_x
        return output.permute(1, 2, 0)

    def predict(self, video_pnr, video_ac, task):
        assert task in ['pnr', 'oscc', 'action_verb', 'action_noun']
        batch_size = video_pnr[0].shape[0]
        encoded_x = self.encode(video_pnr, video_ac)
        y = torch.ones((batch_size, 1)) * self.vocab[task]
        y = y.type_as(video_pnr[0]).long()
        output = self.decode(y, encoded_x)
        if 'action' in task:
            output = torch.argmax(output, dim=-1)
        return output[0, :]

    def predict_ac(self, video_pnr, video_ac):
        batch_size = video_pnr[0].shape[0]
        encoded_x = self.encode(video_pnr, video_ac)
        seq_len = 3  # 3 # set fixed output seq len for now
        output_tokens = (torch.ones((batch_size, seq_len))).type_as(video_pnr[0]).long()
        output_tokens[:, 0] = self.vocab['action']      # start token
        for sy in range(1, seq_len):
            y = output_tokens[:, :sy]
            output = self.decode(y, encoded_x)
            output = torch.argmax(output, dim=-1)
            output_tokens[:, sy] = output[-1, :]
        return output_tokens[:, 1:]  # idx in vocab



class TaskTranslationPromptTransformer6Task(TaskPromptTransformer):
    def __init__(self, args, vocab):
        super(TaskTranslationPromptTransformer6Task, self).__init__(args, vocab)
        self.task_embed = nn.Parameter(torch.randn(1, 4, self.dim), requires_grad=True)
        self.proj_lta = nn.Linear(2048, self.dim)

        cfg_lta = load_lta_config(args.lta_cfg_file)
        self.cfg_lta = copy.deepcopy(cfg_lta)
        cfg_lta.FORECASTING.NUM_ACTIONS_TO_PREDICT = 20
        self.lta_model = ForecastingEncoderDecoder(cfg_lta, build_decoder=False)
        load_lta_backbone(self.lta_model, cfg_lta.CHECKPOINT_FILE_PATH_LTA)
        freeze_params(self.lta_model)


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

    def encode_clips_pnr(self, model, x):
        features = []
        num_inputs = x.shape[1]
        for i in range(num_inputs):
            input_clip = x[:, i, ...]
            tmp = model([input_clip], middle=True)
            features.append(tmp.mean(dim=1))   # average temporal info, to-do: other ways
        return torch.stack(features, dim=1)  #(bs, num_input, 8192)


    def encode(self, video_pnr, video_ac, task):
        with torch.no_grad():
            if 'lta' in task:
                video_oscc = copy.deepcopy(video_pnr)
                feat_pnr = self.encode_clips_pnr(self.pnr_model, video_pnr)
                feat_oscc = self.encode_clips_pnr(self.oscc_model, video_oscc)
                feat_action = self.encode_clips(self.recognition_model, video_ac)
                feat_lta = self.lta_model(video_ac, None, middle=True).transpose(0, 1)  # (bs, 2, d)
            else:
                video_oscc = video_pnr.copy()
                feat_pnr = self.pnr_model(video_pnr, middle=True)
                feat_oscc = self.oscc_model(video_oscc, middle=True)  # (bs, 16, d)
                x_action_list = self.recognition_model(video_ac, middle=True)

        x1 = self.encode_prepare(self.proj_pnr(feat_pnr), 0)
        x2 = self.encode_prepare(self.proj_oscc(feat_oscc), 1)
        if 'lta' in task:
            x3 = self.encode_prepare(feat_action, 2)
            x4 = self.encode_prepare(self.proj_lta(feat_lta), 3)
            x = torch.cat((x1, x2, x3, x4), dim=0)  # (8, bs, 256)
        else:
            feat3_1 = self.proj_action_slow(self.avg_pool_slow(x_action_list[0]).squeeze(-1).squeeze(-1).permute(0, 2, 1))
            feat3_2 = self.proj_action_fast(self.avg_pool_fast(x_action_list[1]).squeeze(-1).squeeze(-1).permute(0, 2, 1))
            x3 = self.encode_prepare(torch.cat((feat3_1, feat3_2), dim=1), 2)
            x = torch.cat((x1, x2, x3), dim=0)    # (48, bs, 256)
        # print(task, x.shape)
        encoded_x = self.transformer_encoder(x)
        return encoded_x

    def forward(self, video_pnr, video_ac, target, task):
        encoded_x = self.encode(video_pnr, video_ac, task)
        output = self.decode(target, encoded_x)
        return output.permute(1, 2, 0)

    def predict(self, video_pnr, video_ac, task, predict_verb_only=False, predict_noun_only=False):
        assert task in ['pnr', 'oscc', 'action', 'lta']
        encoded_x = self.encode(video_pnr, video_ac, task)
        batch_size = encoded_x.shape[1]
        if task in ['action', 'lta']:
            if not predict_noun_only:
                y_verb = torch.ones((batch_size, 1)) * self.vocab[task+'_verb']
                y_verb = y_verb.type_as(video_ac[0]).long()
                output_verb = self.decode(y_verb, encoded_x)

            if predict_verb_only:
                return

            y_noun = torch.ones((batch_size, 1)) * self.vocab[task+'_noun']
            y_noun = y_noun.type_as(video_ac[0]).long()
            output_noun = self.decode(y_noun, encoded_x)

            if predict_noun_only:
                return

            pred_verb = torch.argmax(output_verb, dim=-1)
            pred_noun = torch.argmax(output_noun, dim=-1)
            pred_ac = torch.stack((pred_verb[0, :], pred_noun[0, :]), dim=1)
            return pred_ac

        else:
            y = torch.ones((batch_size, 1)) * self.vocab[task]
            y = y.type_as(video_pnr[0]).long()
            output = self.decode(y, encoded_x)
            return output[0, :]
