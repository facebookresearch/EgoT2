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
from models.pnr.simple_vit import Transformer
from models.pnr.video_model_transfer_3task import TaskFusion3Task
from .video_model_builder import SlowFast
from .head_helper import MultiTaskHead
from .lta_models import ForecastingEncoderDecoder
from utils.pnr.parser import load_config_file as load_pnr_config
from utils.lta.parser import load_config_from_file as load_lta_config
from utils.multitask.load_model import load_checkpoint, load_lta_backbone, freeze_backbone_params, freeze_params

@MODEL_REGISTRY.register()
class Keyframe2Action(TaskFusion3Task):
    def __init__(self, cfg):
        super(Keyframe2Action, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, None, None)
        num_cls1, num_cls2 = cfg.MODEL.NUM_CLASSES
        self.fc1 = nn.Linear(8192, num_cls1)
        self.fc2 = nn.Linear(8192, num_cls2)
        self.act = nn.ReLU()

    def forward(self, x_orig, x_pnr):
        x = self.pnr_model(x_pnr, middle=True).mean(dim=1)
        return [self.fc1(x), self.fc2(x)]


@MODEL_REGISTRY.register()
class State2Action(TaskFusion3Task):
    def __init__(self, cfg):
        super(State2Action, self).__init__(cfg, None, cfg.PRETRAIN.OSCC_CFG, None)
        num_cls1, num_cls2 = cfg.MODEL.NUM_CLASSES
        self.fc1 = nn.Linear(8192, num_cls1)
        self.fc2 = nn.Linear(8192, num_cls2)
        self.act = nn.ReLU()

    def forward(self, x_orig, x_pnr):
        x = self.oscc_model(x_pnr, middle=True).squeeze(1)
        return [self.fc1(x), self.fc2(x)]


@MODEL_REGISTRY.register()
class FinetuneAction(TaskFusion3Task):
    def __init__(self, cfg):
        super(FinetuneAction, self).__init__(cfg, None, None, cfg.PRETRAIN.ACTION_CFG, action_with_head=False)
        num_cls1, num_cls2 = cfg.MODEL.NUM_CLASSES
        self.avg_pool_slow = nn.AdaptiveAvgPool3d((2, 1, 1))
        self.avg_pool_fast = nn.AdaptiveAvgPool3d((16, 1, 1))
        self.fc1 = nn.Linear(8192, num_cls1)
        self.fc2 = nn.Linear(8192, num_cls2)
        self.act = nn.ReLU()

    def forward(self, x_orig):
        x_action_list = self.recognition_model(x_orig, middle=True)  # [(bs, 2048, 8, 7, 7), (bs, 256, 32, 7, 7)]
        bs = x_orig[0].shape[0]
        action_feat_slow = self.avg_pool_slow(x_action_list[0]).reshape(bs, -1)
        action_feat_fast = self.avg_pool_fast(x_action_list[1]).reshape(bs, -1)
        x = torch.cat((action_feat_slow, action_feat_fast), dim=1)
        return [self.fc1(x), self.fc2(x)]


@MODEL_REGISTRY.register()
class TaskFusionLFLinear3TaskSimple(TaskFusion3Task):
    def __init__(self, cfg):
        super(TaskFusionLFLinear3TaskSimple, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, cfg.PRETRAIN.OSCC_CFG, cfg.PRETRAIN.ACTION_CFG, oscc_no_temp_pool=True, action_with_head=True)
        num_cls1, num_cls2 = cfg.MODEL.NUM_CLASSES
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.proj1 = nn.Linear(8192, self.feature_dim)
        self.proj2 = nn.Linear(8192, self.feature_dim)

        self.fc1 = nn.Linear(self.feature_dim * 3, num_cls1)
        self.fc2 = nn.Linear(self.feature_dim * 3, num_cls2)
        self.act = nn.ReLU()

    def forward(self, x_action, x_pnr):
        x_oscc = x_pnr.copy()
        pnr_feat = self.pnr_model(x_pnr, middle=True).mean(dim=1)  # (bs, feat_fim)
        oscc_feat = self.oscc_model(x_oscc, middle=True).mean(dim=1)  # (bs, feat_dim)
        action_feat = self.recognition_model(x_action)

        feat = torch.cat((self.proj1(pnr_feat), self.proj2(oscc_feat), action_feat), dim=1)  # (bs, feat_dim*3)
        out1 = self.fc1(self.act(feat))
        out2 = self.fc2(self.act(feat))
        return [out1, out2]


@MODEL_REGISTRY.register()
class TaskFusionMFTransformer3Task(TaskFusion3Task):
    def __init__(self, cfg):
        super(TaskFusionMFTransformer3Task, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, cfg.PRETRAIN.OSCC_CFG, cfg.PRETRAIN.ACTION_CFG, oscc_no_temp_pool=True, action_with_head=False)
        num_cls1, num_cls2 = cfg.MODEL.NUM_CLASSES
        self.sequence_len = 48
        self.num_heads = cfg.MODEL.TRANSLATION_HEADS
        self.num_layers = cfg.MODEL.TRANSLATION_LAYERS
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES   #256
        self.dp_rate = cfg.MODEL.TRANSLATION_DROPOUT

        self.proj1 = nn.Linear(8192, self.feature_dim)
        self.proj2 = nn.Linear(8192, self.feature_dim)
        self.proj3_slow = nn.Linear(2048, self.feature_dim)
        self.proj3_fast = nn.Linear(256, self.feature_dim)
        self.avg_pool_slow = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avg_pool_fast = nn.AdaptiveAvgPool3d((8, 1, 1))

        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        # self.transformer = Transformer(dim=self.feature_dim, depth=3, heads=8, dim_head=128, mlp_dim=512)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=self.num_heads,
                                                     dropout=self.dp_rate, batch_first=True),
            num_layers=self.num_layers)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.linear_head1 = nn.Sequential(self.ln, nn.Linear(self.feature_dim, num_cls1))
        self.linear_head2 = nn.Sequential(self.ln, nn.Linear(self.feature_dim, num_cls2))

    def forward(self, x_action, x_pnr):
        x_oscc = x_pnr.copy()
        pnr_feat = self.pnr_model(x_pnr, middle=True)  # (bs, 16, feat_fim)
        oscc_feat = self.oscc_model(x_oscc, middle=True)  # (bs, 16, feat_dim)
        x_action_list = self.recognition_model(x_action, middle=True)  # [(bs, 2048, 8, 7, 7), (bs, 256, 32, 7, 7)]
        action_feat_slow = self.avg_pool_slow(x_action_list[0]).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        action_feat_fast = self.avg_pool_fast(x_action_list[1]).squeeze(-1).squeeze(-1).permute(0, 2, 1)

        feat = torch.cat((self.proj3_slow(action_feat_slow), self.proj3_fast(action_feat_fast),
                          self.proj1(pnr_feat), self.proj2(oscc_feat)), dim=1)  # (bs, seq_len, feat_dim)
        feat = self.ln(feat) + self.pe  # (bs, 48, 128)
        out = self.transformer(feat)
        out = out.mean(dim=1)
        return [self.linear_head1(out), self.linear_head2(out)]



@MODEL_REGISTRY.register()
class LTA2Action(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_input = cfg.FORECASTING.NUM_INPUT_CLIPS
        self.input_offset = cfg.FORECASTING.INPUT_OFFSET
        num_cls1, num_cls2 = cfg.MODEL.NUM_CLASSES
        self.feature_dim = 1024
        cfg_lta = load_lta_config(cfg.PRETRAIN.LTA_CFG)
        self.lta_model = ForecastingEncoderDecoder(cfg_lta, build_decoder=False)
        load_lta_backbone(self.lta_model, cfg_lta.CHECKPOINT_FILE_PATH)
        freeze_params(self.lta_model)

        self.fc = nn.Linear(2048 * self.num_input, self.feature_dim)
        self.linear_head1 = nn.Linear(self.feature_dim, num_cls1)
        self.linear_head2 = nn.Linear(self.feature_dim, num_cls2)
        self.act = nn.ReLU()

    def forward(self, x):
        x_lta = [x[0][:, 0:self.num_input, ...], x[1][:, 0:self.num_input, ...]]
        with torch.no_grad():
            feat_lta = self.lta_model(x_lta, middle=True).transpose(0, 1)  # (bs, num_input, 2048)

        bs = feat_lta.shape[0]
        out = self.act(self.fc(feat_lta.reshape(bs, -1)))
        return [self.linear_head1(out), self.linear_head2(out)]


@MODEL_REGISTRY.register()
class TaskFusionMFTransformer2TaskAR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_input = cfg.FORECASTING.NUM_INPUT_CLIPS
        self.input_offset = cfg.FORECASTING.INPUT_OFFSET
        num_cls1, num_cls2 = cfg.MODEL.NUM_CLASSES
        self.sequence_len = 18
        self.num_heads = cfg.MODEL.TRANSLATION_HEADS
        self.num_layers = cfg.MODEL.TRANSLATION_LAYERS
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.dp_rate = cfg.MODEL.TRANSLATION_DROPOUT

        self.proj_lta = nn.Linear(2048, self.feature_dim)
        self.proj_slow = nn.Linear(2048, self.feature_dim)
        self.proj_fast = nn.Linear(256, self.feature_dim)
        self.avg_pool_slow = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avg_pool_fast = nn.AdaptiveAvgPool3d((8, 1, 1))

        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=self.num_heads,
                                                     dropout=self.dp_rate, batch_first=True),
            num_layers=self.num_layers)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.linear_head1 = nn.Sequential(self.ln, nn.Linear(self.feature_dim, num_cls1))
        self.linear_head2 = nn.Sequential(self.ln, nn.Linear(self.feature_dim, num_cls2))
        self._init_parameters()

        cfg_recognition = load_lta_config(cfg.PRETRAIN.ACTION_CFG)
        cfg_recognition.MODEL.NUM_CLASSES = [self.feature_dim]
        cfg_recognition.MODEL.HEAD_ACT = None

        self.action_model = SlowFast(cfg_recognition, with_head=False)
        load_lta_backbone(self.action_model, cfg_recognition.CHECKPOINT_FILE_PATH, True, True)
        freeze_backbone_params(self.action_model)

        cfg_lta = load_lta_config(cfg.PRETRAIN.LTA_CFG)
        self.lta_model = ForecastingEncoderDecoder(cfg_lta, build_decoder=False)
        load_lta_backbone(self.lta_model, cfg_lta.CHECKPOINT_FILE_PATH)
        freeze_params(self.lta_model)


    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x):
        x1 = x.copy()
        x_action = [x1[0][:, -1, ...], x1[1][:, -1, ...]]
        x_lta = [x[0][:, 0:self.num_input, ...], x[1][:, 0:self.num_input, ...]]
        with torch.no_grad():
            x_action_list = self.action_model(x_action, middle=True)  # [(bs, 2048, 8, 7, 7), (bs, 256, 32, 7, 7)]
            feat_lta = self.lta_model(x_lta, middle=True).transpose(0, 1)  # (bs, num_input, 2048)

        action_feat_slow = self.avg_pool_slow(x_action_list[0]).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        action_feat_fast = self.avg_pool_fast(x_action_list[1]).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        feat = torch.cat((self.proj_slow(action_feat_slow), self.proj_fast(action_feat_fast),
                          self.proj_lta(feat_lta)), dim=1)
        feat = self.ln(feat) + self.pe  # (bs, 16+2, dim)
        out = self.transformer(feat)
        out = out.mean(dim=1)
        return [self.linear_head1(out), self.linear_head2(out)]

