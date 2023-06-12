#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import copy
import math
from functools import reduce
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from .build import MODEL_REGISTRY
from models.pnr.video_model_builder import KeyframeLocalizationResNet, StateChangeClsResNet
from .video_model_builder import SlowFast
from .head_helper import MultiTaskHead
from .lta_models import ForecastingEncoderDecoder
from utils.pnr.parser import load_config_file as load_pnr_config
from utils.multitask.load_model import load_ckpt, load_lta_backbone, freeze_backbone_params, freeze_params


@MODEL_REGISTRY.register()
class State2LTA(nn.Module):
    def __init__(self, cfg):
        super(State2LTA, self).__init__()
        self.cfg = cfg
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.proj_oscc = nn.Linear(8192, self.feature_dim)

        cfg_oscc = load_pnr_config(cfg.PRETRAIN.OSCC_CFG)
        self.cfg_pnr = cfg_oscc
        cfg_oscc.MODEL.NO_TEMP_POOL = False
        self.oscc_model = StateChangeClsResNet(cfg_oscc)
        load_ckpt(self.oscc_model, cfg_oscc.MISC.CHECKPOINT_FILE_PATH)
        freeze_params(self.oscc_model)

        head_classes = [reduce((lambda x, y: x + y),
                               cfg.MODEL.NUM_CLASSES)] * self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        self.head = MultiTaskHead(
            dim_in=[self.feature_dim * cfg.FORECASTING.NUM_INPUT_CLIPS],
            num_classes=head_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )

    def encode_clips_pnr(self, model, x):
        features = []
        num_inputs = x.shape[1]
        for i in range(num_inputs):
            input_clip = x[:, i, ...]
            tmp = model([input_clip], middle=True)
            features.append(tmp.mean(dim=1))   # average temporal info, to-do: other ways
        return torch.stack(features, dim=1)  #(bs, num_input, 8192)

    def decode(self, x):
        x = x.view(x.shape[0], -1, 1, 1, 1)
        x = torch.stack(self.head([x]), dim=1)  # (B, Z, #verbs + #nouns)
        x = torch.split(x, self.cfg.MODEL.NUM_CLASSES, dim=-1)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return x

    def forward(self, x_lta, x_pnr):
        x = self.proj_oscc(self.encode_clips_pnr(self.oscc_model, x_pnr))
        feat = x.view(x.shape[0], -1)
        return self.decode(feat)


    def generate(self, x_lta, x_pnr, k=1):
        x = self.forward(x_lta, x_pnr)
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
class Keyframe2LTA(nn.Module):
    def __init__(self, cfg):
        super(Keyframe2LTA, self).__init__()
        self.cfg = cfg
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.proj_pnr = nn.Linear(8192, self.feature_dim)

        cfg_pnr = load_pnr_config(cfg.PRETRAIN.PNR_CFG)
        self.cfg_pnr = cfg_pnr
        self.pnr_model = KeyframeLocalizationResNet(cfg_pnr)
        load_ckpt(self.pnr_model, cfg_pnr.MISC.CHECKPOINT_FILE_PATH)
        freeze_params(self.pnr_model)

        head_classes = [reduce((lambda x, y: x + y),
                               cfg.MODEL.NUM_CLASSES)] * self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        self.head = MultiTaskHead(
            dim_in=[self.feature_dim * cfg.FORECASTING.NUM_INPUT_CLIPS],
            num_classes=head_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )

    def encode_clips_pnr(self, model, x):
        features = []
        num_inputs = x.shape[1]
        for i in range(num_inputs):
            input_clip = x[:, i, ...]
            tmp = model([input_clip], middle=True)
            features.append(tmp.mean(dim=1))   # average temporal info, to-do: other ways
        return torch.stack(features, dim=1)  #(bs, num_input, 8192)

    def decode(self, x):
        x = x.view(x.shape[0], -1, 1, 1, 1)
        x = torch.stack(self.head([x]), dim=1)  # (B, Z, #verbs + #nouns)
        x = torch.split(x, self.cfg.MODEL.NUM_CLASSES, dim=-1)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return x

    def forward(self, x_lta, x_pnr):
        x = self.proj_pnr(self.encode_clips_pnr(self.pnr_model, x_pnr))
        feat = x.view(x.shape[0], -1)
        return self.decode(feat)


    def generate(self, x_lta, x_pnr, k=1):
        x = self.forward(x_lta, x_pnr)
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
class TaskFusionLFLinear4Task(nn.Module):
    def __init__(self, cfg):
        super(TaskFusionLFLinear4Task, self).__init__()
        self.cfg = cfg
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES

        self.proj_pnr = nn.Linear(8192, self.feature_dim)
        self.proj_oscc = nn.Linear(8192, self.feature_dim)
        self.proj_lta = nn.Linear(2048, self.feature_dim)
        self.fc = nn.Linear(4 * self.feature_dim, self.feature_dim)
        self.act = nn.ReLU()

        # Load the four task-specfic models
        cfg_pnr = load_pnr_config(cfg.PRETRAIN.PNR_CFG)
        self.cfg_pnr = cfg_pnr
        self.pnr_model = KeyframeLocalizationResNet(cfg_pnr)
        load_ckpt(self.pnr_model, cfg_pnr.MISC.CHECKPOINT_FILE_PATH)
        freeze_params(self.pnr_model)

        cfg_oscc = load_pnr_config(cfg.PRETRAIN.OSCC_CFG)
        cfg_oscc.MODEL.NO_TEMP_POOL = False
        self.oscc_model = StateChangeClsResNet(cfg_oscc)
        load_ckpt(self.oscc_model, cfg_oscc.MISC.CHECKPOINT_FILE_PATH)
        freeze_params(self.oscc_model)

        backbone_cfg = copy.deepcopy(cfg)
        backbone_cfg.MODEL.NUM_CLASSES = [self.feature_dim]
        backbone_cfg.MODEL.HEAD_ACT = None
        self.action_model = SlowFast(backbone_cfg, with_head=True)
        load_lta_backbone(self.action_model, cfg.CHECKPOINT_FILE_PATH_AR, True, True)
        freeze_backbone_params(self.action_model)

        self.lta_model = ForecastingEncoderDecoder(cfg, build_decoder=False)
        load_lta_backbone(self.lta_model, cfg.CHECKPOINT_FILE_PATH_LTA)
        freeze_params(self.lta_model)

        # Decoder
        head_classes = [reduce((lambda x, y: x + y),
                               cfg.MODEL.NUM_CLASSES)] * self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        self.head = MultiTaskHead(
            dim_in=[self.feature_dim],
            num_classes=head_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


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

    def decode(self, x):
        x = x.view(x.shape[0], -1, 1, 1, 1)
        x = torch.stack(self.head([x]), dim=1)  # (B, Z, #verbs + #nouns)
        x = torch.split(x, self.cfg.MODEL.NUM_CLASSES, dim=-1)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return x

    def forward(self, x_lta, x_pnr):
        feat_pnr = self.proj_pnr(self.encode_clips_pnr(self.pnr_model, x_pnr)).mean(dim=1)
        feat_oscc = self.proj_oscc(self.encode_clips_pnr(self.oscc_model, x_pnr)).mean(dim=1)
        feat_action = self.encode_clips(self.action_model, x_lta).mean(dim=1)
        feat_lta = self.proj_lta(self.lta_model(x_lta, None, middle=True).transpose(0, 1)).mean(dim=1) # (bs, num_input, d)
        feat = torch.cat((feat_pnr, feat_oscc, feat_action, feat_lta), dim=1) # (bs, 4*d)
        out = self.act(self.fc(self.act(feat)))
        return self.decode(out)

    def generate(self, x_lta, x_pnr, k=1):
        x = self.forward(x_lta, x_pnr)
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
class TaskFusionMFTransformerLTA4Task(nn.Module):
    def __init__(self, cfg):
        super(TaskFusionMFTransformerLTA4Task, self).__init__()
        self.cfg = cfg
        self.sequence_len = cfg.FORECASTING.NUM_INPUT_CLIPS * 4
        self.num_heads = cfg.MODEL.TRANSLATION_HEADS
        self.num_layers = cfg.MODEL.TRANSLATION_LAYERS
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.dp_rate = cfg.MODEL.TRANSLATION_DROPOUT
        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        self.proj_pnr = nn.Linear(8192, self.feature_dim)
        self.proj_oscc = nn.Linear(8192, self.feature_dim)
        self.proj_lta = nn.Linear(2048, self.feature_dim)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=self.num_heads,
                                                     dropout=self.dp_rate, batch_first=True),
            num_layers=self.num_layers)
        self.ln = nn.LayerNorm(self.feature_dim)
        self._init_parameters()

        # Load the four task-specfic models
        cfg_pnr = load_pnr_config(cfg.PRETRAIN.PNR_CFG)
        self.cfg_pnr = cfg_pnr
        self.pnr_model = KeyframeLocalizationResNet(cfg_pnr)
        load_ckpt(self.pnr_model, cfg_pnr.MISC.CHECKPOINT_FILE_PATH)
        freeze_params(self.pnr_model)

        cfg_oscc = load_pnr_config(cfg.PRETRAIN.OSCC_CFG)
        cfg_oscc.MODEL.NO_TEMP_POOL = False
        self.oscc_model = StateChangeClsResNet(cfg_oscc)
        load_ckpt(self.oscc_model, cfg_oscc.MISC.CHECKPOINT_FILE_PATH)
        freeze_params(self.oscc_model)

        backbone_cfg = copy.deepcopy(cfg)
        backbone_cfg.MODEL.NUM_CLASSES = [self.feature_dim]
        backbone_cfg.MODEL.HEAD_ACT = None
        self.action_model = SlowFast(backbone_cfg, with_head=True)
        load_lta_backbone(self.action_model, cfg.CHECKPOINT_FILE_PATH_AR, True, True)
        freeze_backbone_params(self.action_model)

        self.lta_model = ForecastingEncoderDecoder(cfg, build_decoder=True) #False
        load_lta_backbone(self.lta_model, cfg.CHECKPOINT_FILE_PATH_LTA)
        freeze_params(self.lta_model)

        # Decoder
        head_classes = [reduce((lambda x, y: x + y),
                               cfg.MODEL.NUM_CLASSES)] * self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        self.head = MultiTaskHead(
            dim_in=[self.feature_dim],
            num_classes=head_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


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

    def decode(self, x):
        x = x.view(x.shape[0], -1, 1, 1, 1)
        x = torch.stack(self.head([x]), dim=1)  # (B, Z, #verbs + #nouns)
        x = torch.split(x, self.cfg.MODEL.NUM_CLASSES, dim=-1)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return x

    def forward(self, x_lta, x_pnr):
        feat_pnr = self.proj_pnr(self.encode_clips_pnr(self.pnr_model, x_pnr))
        feat_oscc = self.proj_oscc(self.encode_clips_pnr(self.oscc_model, x_pnr))
        feat_action = self.encode_clips(self.action_model, x_lta)
        feat_lta = self.proj_lta(self.lta_model(x_lta, None, middle=True).transpose(0, 1)) # (bs, num_input, d)
        feat = torch.cat((feat_pnr, feat_oscc, feat_action, feat_lta), dim=1)
        feat = self.ln(feat) + self.pe
        out = self.transformer(feat)
        out = out.mean(dim=1)
        return self.decode(out)

    def generate(self, x_lta, x_pnr, k=1):
        x = self.forward(x_lta, x_pnr)
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
class FinetuneLTA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lta_model = ForecastingEncoderDecoder(cfg, build_decoder=False)
        load_lta_backbone(self.lta_model, cfg.CHECKPOINT_FILE_PATH_LTA)
        freeze_params(self.lta_model)

        head_classes = [reduce((lambda x, y: x + y),
                               cfg.MODEL.NUM_CLASSES)] * self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        self.head = MultiTaskHead(
            dim_in=[2048 * 2],
            num_classes=head_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )

    def decode(self, x):
        x = x.view(x.shape[0], -1, 1, 1, 1)
        x = torch.stack(self.head([x]), dim=1)  # (B, Z, #verbs + #nouns)
        x = torch.split(x, self.cfg.MODEL.NUM_CLASSES, dim=-1)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return x

    def forward(self, x, tgts=None):
        feat_lta = self.lta_model(x, None, middle=True).transpose(0, 1)  # (bs, num_inputs, d)
        bs = feat_lta.shape[0]
        x = feat_lta.reshape(bs, -1)
        out = self.decode(x)
        return out
    
    def generate(self, x, k=1):
        x = self.forward(x)
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
class TaskFusionMFTransformer2Task(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sequence_len = cfg.FORECASTING.NUM_INPUT_CLIPS * 2
        self.num_heads = cfg.MODEL.TRANSLATION_HEADS
        self.num_layers = cfg.MODEL.TRANSLATION_LAYERS
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.proj_lta = nn.Identity()
        if self.feature_dim != 2048:
            self.proj_lta = nn.Linear(2048, self.feature_dim)
        self.dp_rate = cfg.MODEL.TRANSLATION_DROPOUT
        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        # self.transformer = Transformer(dim=self.feature_dim, depth=self.num_layers, heads=self.num_heads, dim_head=self.feature_dim, mlp_dim=self.feature_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=self.num_heads,
                                                     dropout=self.dp_rate, batch_first=True),
            num_layers=self.num_layers)
        self.ln = nn.LayerNorm(self.feature_dim)
        self._init_parameters()

        backbone_cfg = copy.deepcopy(cfg)
        backbone_cfg.MODEL.NUM_CLASSES = [self.feature_dim]
        backbone_cfg.MODEL.HEAD_ACT = None

        self.action_model = SlowFast(backbone_cfg, with_head=True)
        load_lta_backbone(self.action_model, cfg.CHECKPOINT_FILE_PATH_AR, True, True)
        freeze_backbone_params(self.action_model)

        self.lta_model = ForecastingEncoderDecoder(cfg, build_decoder=False)
        load_lta_backbone(self.lta_model, cfg.CHECKPOINT_FILE_PATH_LTA)
        freeze_params(self.lta_model)

        head_classes = [reduce((lambda x, y: x + y), cfg.MODEL.NUM_CLASSES)] * self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        self.head = MultiTaskHead(
            dim_in=[self.feature_dim],
            num_classes=head_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    def decode(self, x):
        x = x.view(x.shape[0], -1, 1, 1, 1)
        x = torch.stack(self.head([x]), dim=1)  # (B, Z, #verbs + #nouns)
        x = torch.split(x, self.cfg.MODEL.NUM_CLASSES, dim=-1)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return x

    def forward(self, x, tgts=None):
        feat_action = self.encode_clips(self.action_model, x)  # (bs, num_input, d)
        feat_lta = self.lta_model(x, None, middle=True).transpose(0, 1)
        # feat_lta = self.encode_clips(self.lta_model, x)
        feat = torch.cat((feat_action, self.proj_lta(feat_lta)), dim=1)   # (bs, num_input*2, d)
        feat = self.ln(feat) + self.pe
        out = self.transformer(feat)
        out = out.mean(dim=1)
        out = self.decode(out)
        # print(out[0].shape, out[1].shape)
        return out

    def generate(self, x, k=1):
        x = self.forward(x)
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


from .lta_models_seqdecoder import CustomDecoderLayer, PositionalEncoding
from utils.multitask.build_vocab import vocab_idx_to_orig
@MODEL_REGISTRY.register()
class TaskFusionMFTransformer2TaskSeqDecoder(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.v_idx, self.n_idx = vocab_idx_to_orig()
        self.sequence_len = cfg.FORECASTING.NUM_INPUT_CLIPS * 2
        self.num_heads = cfg.MODEL.TRANSLATION_HEADS
        self.num_layers = cfg.MODEL.TRANSLATION_LAYERS
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.dp_rate = cfg.MODEL.TRANSLATION_DROPOUT

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=self.num_heads, dropout=self.dp_rate),
            num_layers=self.num_layers)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=CustomDecoderLayer(d_model=self.feature_dim, nhead=self.num_heads, dropout=self.dp_rate),
            num_layers=self.num_layers
        )

        self.ln = nn.LayerNorm(self.feature_dim)
        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        self.pos_embed = PositionalEncoding(self.feature_dim, dropout=self.dp_rate)
        self.embedding = nn.Embedding(len(self.vocab), self.feature_dim)
        self.y_mask = self.get_tgt_mask(3)
        self.fc = nn.Linear(self.feature_dim, len(self.vocab))

        self._init_parameters()

        backbone_cfg = copy.deepcopy(cfg)
        backbone_cfg.MODEL.NUM_CLASSES = [self.feature_dim]
        backbone_cfg.MODEL.HEAD_ACT = None

        self.action_model = SlowFast(backbone_cfg, with_head=True)
        load_lta_backbone(self.action_model, cfg.CHECKPOINT_FILE_PATH_AR, True, True)
        freeze_backbone_params(self.action_model)

        # self.lta_model = SlowFast(backbone_cfg, with_head=True)
        # load_lta_backbone(self.lta_model, cfg.CHECKPOINT_FILE_PATH_LTA)
        # freeze_backbone_params(self.lta_model)
        self.lta_model = ForecastingEncoderDecoder(cfg)
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

    def encode(self, x):
        feat_action = self.encode_clips(self.action_model, x)  # (bs, num_input, d)
        feat_lta = self.lta_model(x, None, middle=True).transpose(0, 1)
        feat = torch.cat((feat_action, feat_lta), dim=1)    # (bs, num_input*2, d)
        feat = (self.ln(feat) + self.pe).transpose(0, 1)
        return self.transformer_encoder(feat)   # (num_inputs*2, bs, dim)

    def decode(self, y, encoded_x):
        sy = y.size(1)
        y = y.permute(1, 0)  # (bs, seq_y)->(seq_y, bs)
        y = self.embedding(y) * math.sqrt(self.feature_dim)  # (seq_y, bs, dim)
        y = self.pos_embed(y)
        # y_mask = self.get_tgt_mask(sequence_length).type_as(encoded_x)
        y_mask = self.y_mask[:sy, :sy].type_as(encoded_x)
        output = self.transformer_decoder(y, encoded_x, y_mask)
        output = self.fc(output)  # (seq_y, bs, vocab_size)
        return output

    def forward(self, x, target):
        encoded_x = self.encode(x)
        output = self.decode(target, encoded_x)
        return output.permute(1, 2, 0)

    def predict(self, x, only_verb=False):
        encoded_x = self.encode(x)
        batch_size = encoded_x.shape[1]
        y_verb = torch.ones((batch_size, 1)) * self.vocab['lta_verb']
        y_verb = y_verb.type_as(x[0]).long()
        output_verb = self.decode(y_verb, encoded_x)
        preds_verb = output_verb[0, :, self.v_idx].unsqueeze(dim=1)
        if only_verb:
            return

        y_noun = torch.ones((batch_size, 1)) * self.vocab['lta_noun']
        y_noun = y_noun.type_as(x[0]).long()
        output_noun = self.decode(y_noun, encoded_x)
        preds_noun = output_noun[0, :, self.n_idx].unsqueeze(dim=1)

        return [preds_verb, preds_noun]

    def generate(self, x, k=1):
        x = self.predict(x)
        results = []
        for head_x in x:
            if k > 1:
                preds_dist = Categorical(logits=head_x)
                preds = [preds_dist.sample() for _ in range(k)]
            elif k == 1:
                preds = [head_x.argmax(2)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)

        return results
