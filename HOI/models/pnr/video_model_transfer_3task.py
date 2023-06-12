#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

"""Video models."""
import torch
import torch.nn as nn
from models.lta.video_model_builder import SlowFast

from .simple_vit import Transformer
from .video_model_builder import KeyframeLocalizationResNet, StateChangeClsResNet
from utils.pnr.parser import load_config_file
from utils.lta.parser import load_config_from_file as load_lta_config
from utils.lta.parser import parse_args
from utils.multitask.load_model import load_checkpoint, freeze_params, load_recognition_backbone, freeze_backbone_params
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class TaskFusion3Task(nn.Module):
    def __init__(self, cfg, cfg_pnr_file=None, cfg_oscc_file=None, cfg_recognition_file=None, oscc_no_temp_pool=False, action_with_head=True):
        super(TaskFusion3Task, self).__init__()
        self.cfg_pnr = None
        self.cfg_recognition = None
        if cfg_pnr_file:
            cfg_pnr = load_config_file(cfg_pnr_file)
            self.cfg_pnr = cfg_pnr
            self.pnr_model = KeyframeLocalizationResNet(cfg_pnr)
            load_checkpoint(self.pnr_model, cfg_pnr.MISC.CHECKPOINT_FILE_PATH)
            if cfg.PRETRAIN.PNR_FT:
                self.pnr_model.eval()
                freeze_params(self.pnr_model)

        if cfg_oscc_file:
            cfg_oscc = load_config_file(cfg_oscc_file)
            self.cfg_pnr = cfg_oscc
            cfg_oscc.MODEL.NO_TEMP_POOL = oscc_no_temp_pool
            self.oscc_model = StateChangeClsResNet(cfg_oscc)
            load_checkpoint(self.oscc_model, cfg_oscc.MISC.CHECKPOINT_FILE_PATH)
            if cfg.PRETRAIN.OSCC_FT:
                self.oscc_model.eval()
                freeze_params(self.oscc_model)

        if cfg_recognition_file:
            # args = parse_args()
            # args.cfg_file = cfg_recognition_file
            cfg_recognition = load_lta_config(cfg_recognition_file)
            cfg_recognition.MODEL.NUM_CLASSES = [cfg.MODEL.TRANSLATION_INPUT_FEATURES]
            cfg_recognition.MODEL.HEAD_ACT = None
            self.cfg_recognition = cfg_recognition

            self.recognition_model = SlowFast(cfg_recognition, with_head=action_with_head)
            load_recognition_backbone(self.recognition_model, cfg_recognition.CHECKPOINT_FILE_PATH)
            if cfg.PRETRAIN.ACTION_FT:
                self.recognition_model.eval()
                freeze_backbone_params(self.recognition_model)  # do not freeze head


    def forward(self, x1, x2):
        raise NotImplementedError



@MODEL_REGISTRY.register()
class TaskFusionLFLinear3TaskSimple(TaskFusion3Task):
    def __init__(self, cfg):
        super(TaskFusionLFLinear3TaskSimple, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, cfg.PRETRAIN.OSCC_CFG, cfg.PRETRAIN.ACTION_CFG, oscc_no_temp_pool=True, action_with_head=True)
        self.num_classes = 16 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.unsqueeze_dim = 1 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.proj1 = nn.Linear(8192, self.feature_dim)
        self.proj2 = nn.Linear(8192, self.feature_dim)

        self.fc1 = nn.Linear(self.feature_dim * 3, self.num_classes)
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        x_pnr = x1
        x_oscc = x1.copy()
        x_action = x2

        pnr_feat = self.pnr_model(x_pnr, middle=True).mean(dim=1)  # (bs, 16, feat_fim)
        oscc_feat = self.oscc_model(x_oscc, middle=True).mean(dim=1)  # (bs, 16, feat_dim)
        action_feat = self.recognition_model(x_action)

        feat = torch.cat((self.proj1(pnr_feat), self.proj2(oscc_feat), action_feat), dim=1)
        out = self.fc1(self.act(feat))
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class TaskFusionLFLinear3Task(TaskFusion3Task):
    def __init__(self, cfg):
        super(TaskFusionLFLinear3Task, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, cfg.PRETRAIN.OSCC_CFG, cfg.PRETRAIN.ACTION_CFG, oscc_no_temp_pool=True, action_with_head=False)
        self.num_classes = 16 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.unsqueeze_dim = 1 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.feature_dim = 512
        self.proj1 = nn.Linear(8192, self.feature_dim)
        self.proj2 = nn.Linear(8192, self.feature_dim)
        self.proj3_slow = nn.Linear(2048, self.feature_dim)
        self.proj3_fast = nn.Linear(256, self.feature_dim)
        self.avg_pool_slow = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avg_pool_fast = nn.AdaptiveAvgPool3d((8, 1, 1))

        self.fc1 = nn.Linear(self.feature_dim, self.num_classes)
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        x_pnr = x1
        x_oscc = x1.copy()
        x_action = x2

        pnr_feat = self.pnr_model(x_pnr, middle=True)  # (bs, 16, feat_fim)
        oscc_feat = self.oscc_model(x_oscc, middle=True)  # (bs, 16, feat_dim)
        x_action_list = self.recognition_model(x_action, middle=True)  # [(bs, 2048, 8, 7, 7), (bs, 256, 32, 7, 7)]
        action_feat_slow = self.avg_pool_slow(x_action_list[0]).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        action_feat_fast = self.avg_pool_fast(x_action_list[1]).squeeze(-1).squeeze(-1).permute(0, 2, 1)

        feat = torch.cat((self.proj1(pnr_feat), self.proj2(oscc_feat), self.proj3_slow(action_feat_slow), self.proj3_fast(action_feat_fast)), dim=1)  # (bs, seq_len, feat_dim)
        feat = feat.mean(dim=1)  # (bs, feat_dim)
        out = self.fc1(self.act(feat))
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class TaskFusionMFTransformer3Task(TaskFusion3Task):
    """mid fusion transformer"""
    def __init__(self, cfg):
        super(TaskFusionMFTransformer3Task, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, cfg.PRETRAIN.OSCC_CFG, cfg.PRETRAIN.ACTION_CFG, oscc_no_temp_pool=True, action_with_head=False)
        self.num_classes = 16 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.unsqueeze_dim = 1 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.sequence_len = 48
        self.feature_dim = 256
        self.proj1 = nn.Linear(8192, self.feature_dim)
        self.proj2 = nn.Linear(8192, self.feature_dim)
        self.proj3_slow = nn.Linear(2048, self.feature_dim)
        self.proj3_fast = nn.Linear(256, self.feature_dim)
        self.avg_pool_slow = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avg_pool_fast = nn.AdaptiveAvgPool3d((8, 1, 1))
        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        self.transformer = Transformer(dim=self.feature_dim, depth=3, heads=8, dim_head=128, mlp_dim=512)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.linear_head = nn.Sequential(self.ln, nn.Linear(self.feature_dim, self.num_classes))

    def forward(self, x1, x2):
        x_pnr = x1
        x_oscc = x1.copy()
        x_action = x2

        pnr_feat = self.pnr_model(x_pnr, middle=True)  # (bs, 16, feat_fim)
        oscc_feat = self.oscc_model(x_oscc, middle=True)  # (bs, 16, feat_dim)
        x_action_list = self.recognition_model(x_action, middle=True)  # [(bs, 2048, 8, 7, 7), (bs, 256, 32, 7, 7)]
        action_feat_slow = self.avg_pool_slow(x_action_list[0]).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        action_feat_fast = self.avg_pool_fast(x_action_list[1]).squeeze(-1).squeeze(-1).permute(0, 2, 1)

        feat = torch.cat((self.proj1(pnr_feat), self.proj2(oscc_feat), self.proj3_slow(action_feat_slow), self.proj3_fast(action_feat_fast)), dim=1)  # (bs, seq_len, feat_dim)
        feat = self.ln(feat) + self.pe # (bs, 48, 128)
        out = self.transformer(feat)
        out = out.mean(dim=1)
        out = self.linear_head(out)
        return out.unsqueeze(self.unsqueeze_dim)



@MODEL_REGISTRY.register()
class TaskFusionLFTransformer3TaskDropout(TaskFusion3Task):
    """late fusion transformer"""
    def __init__(self, cfg):
        super(TaskFusionLFTransformer3TaskDropout, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, cfg.PRETRAIN.OSCC_CFG, cfg.PRETRAIN.ACTION_CFG, oscc_no_temp_pool=True, action_with_head=True)
        self.num_classes = 16 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.unsqueeze_dim = 1 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.sequence_len = 3
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.num_layers = cfg.MODEL.TRANSLATION_LAYERS
        self.proj1 = nn.Linear(8192, self.feature_dim)
        self.proj2 = nn.Linear(8192, self.feature_dim)
        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.dp = nn.Dropout(cfg.MODEL.FEAT_DROPOUT_RATE)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8,
                                                     dropout=cfg.MODEL.TRANSFORMER_DROPOUT_RATE,
                                                     dim_feedforward=self.feature_dim * 2, batch_first=True),
            num_layers=self.num_layers)
        self.linear_head = nn.Sequential(self.ln, nn.Linear(self.feature_dim, self.num_classes))

    def forward(self, x1, x2):
        x_pnr = x1
        x_oscc = x1.copy()
        x_action = x2

        pnr_feat = self.pnr_model(x_pnr, middle=True).mean(dim=1)  # (bs, 16, feat_fim)
        oscc_feat = self.oscc_model(x_oscc, middle=True).mean(dim=1)  # (bs, 16, feat_dim)
        action_feat = self.recognition_model(x_action)

        pnr_feat = self.dp(self.proj1(pnr_feat))
        oscc_feat = self.dp(self.proj2(oscc_feat))
        action_feat = self.dp(action_feat)

        feat = torch.stack((pnr_feat, oscc_feat, action_feat), dim=1)  # (bs, 3, feat_dim)
        feat = self.ln(feat) + self.pe # (bs, 3, 128)
        out = self.transformer(feat)
        out = out.mean(dim=1)
        out = self.linear_head(out)
        return out.unsqueeze(self.unsqueeze_dim)



@MODEL_REGISTRY.register()
class TaskFusionMFTransformer3TaskDropout(TaskFusion3Task):
    """mid fusion transformer"""
    def __init__(self, cfg):
        super(TaskFusionMFTransformer3TaskDropout, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, cfg.PRETRAIN.OSCC_CFG, cfg.PRETRAIN.ACTION_CFG, oscc_no_temp_pool=True, action_with_head=False)
        self.num_classes = 16 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.unsqueeze_dim = 1 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.sequence_len = 48
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.num_layers = cfg.MODEL.TRANSLATION_LAYERS
        self.proj1 = nn.Linear(8192, self.feature_dim)
        self.proj2 = nn.Linear(8192, self.feature_dim)
        self.proj3_slow = nn.Linear(2048, self.feature_dim)
        self.proj3_fast = nn.Linear(256, self.feature_dim)
        self.avg_pool_slow = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avg_pool_fast = nn.AdaptiveAvgPool3d((8, 1, 1))
        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.dp = nn.Dropout(cfg.MODEL.FEAT_DROPOUT_RATE)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8,
                                                     dropout=cfg.MODEL.TRANSFORMER_DROPOUT_RATE,
                                                     dim_feedforward=self.feature_dim * 2, batch_first=True),
            num_layers=self.num_layers)
        self.linear_head = nn.Sequential(self.ln, nn.Linear(self.feature_dim, self.num_classes))

    def forward(self, x1, x2):
        x_pnr = x1
        x_oscc = x1.copy()
        x_action = x2

        pnr_feat = self.pnr_model(x_pnr, middle=True)  # (bs, 16, feat_fim)
        oscc_feat = self.oscc_model(x_oscc, middle=True)  # (bs, 16, feat_dim)
        x_action_list = self.recognition_model(x_action, middle=True)  # [(bs, 2048, 8, 7, 7), (bs, 256, 32, 7, 7)]
        action_feat_slow = self.avg_pool_slow(x_action_list[0]).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        action_feat_fast = self.avg_pool_fast(x_action_list[1]).squeeze(-1).squeeze(-1).permute(0, 2, 1)

        pnr_feat = self.dp(self.proj1(pnr_feat))
        oscc_feat = self.dp(self.proj2(oscc_feat))
        action_feat1 = self.dp(self.proj3_slow(action_feat_slow))
        action_feat2 = self.dp(self.proj3_fast(action_feat_fast))
        feat = torch.cat((pnr_feat, oscc_feat, action_feat1, action_feat2), dim=1)  # (bs, seq_len, feat_dim)
        feat = self.ln(feat) + self.pe # (bs, 48, 128)
        out = self.transformer(feat)
        out = out.mean(dim=1)
        out = self.linear_head(out)
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class Action2State(TaskFusion3Task):
    """transfer lr: from action recognition -> oscc"""
    def __init__(self, cfg):
        super(Action2State, self).__init__(cfg, None, None, cfg.PRETRAIN.ACTION_CFG)
        self.num_classes = 2
        self.unsqueeze_dim = 2
        self.in_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.fc1 = nn.Linear(self.in_dim, self.num_classes)
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        x = self.act(self.recognition_model(x2))
        out = self.fc1(x)
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class Action2Keyframe(TaskFusion3Task):
    """transfer lr: from oscc -> pnr"""
    def __init__(self, cfg):
        super(Action2Keyframe, self).__init__(cfg, None, None, cfg.PRETRAIN.ACTION_CFG)
        self.num_classes = 16
        self.unsqueeze_dim = 1
        self.in_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.fc1 = nn.Linear(self.in_dim, self.num_classes)
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        x = self.act(self.recognition_model(x2))
        out = self.fc1(x)
        return out.unsqueeze(self.unsqueeze_dim)