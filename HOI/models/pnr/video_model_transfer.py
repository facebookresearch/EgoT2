#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

"""Video models."""
import torch
import torch.nn as nn
from .simple_vit import Transformer
from .video_model_builder import KeyframeLocalizationResNet, StateChangeClsResNet
from utils.pnr.parser import load_config_file
from utils.multitask.load_model import load_checkpoint, freeze_params
from .build import MODEL_REGISTRY


class TaskFusion(nn.Module):
    def __init__(self, cfg, cfg_pnr_file=None, cfg_oscc_file=None, oscc_no_temp_pool=False):
        super(TaskFusion, self).__init__()
        if cfg_pnr_file:
            cfg_pnr = load_config_file(cfg_pnr_file)
            self.pnr_model = KeyframeLocalizationResNet(cfg_pnr)
            load_checkpoint(self.pnr_model, cfg_pnr.MISC.CHECKPOINT_FILE_PATH)
            if cfg.PRETRAIN.PNR_FT:
                self.pnr_model.eval()
                freeze_params(self.pnr_model)

        if cfg_oscc_file:
            cfg_oscc = load_config_file(cfg_oscc_file)
            cfg_oscc.MODEL.NO_TEMP_POOL = oscc_no_temp_pool
            self.oscc_model = StateChangeClsResNet(cfg_oscc)
            load_checkpoint(self.oscc_model, cfg_oscc.MISC.CHECKPOINT_FILE_PATH)
            if cfg.PRETRAIN.OSCC_FT:
                self.oscc_model.eval()
                freeze_params(self.oscc_model)

        self.cfg_recognition = None

    def forward(self, x):
        raise NotImplementedError


@MODEL_REGISTRY.register()
class TaskFusionMFTransformer(TaskFusion):
    """mid fusion transformer"""
    def __init__(self, cfg):
        super(TaskFusionMFTransformer, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, cfg.PRETRAIN.OSCC_CFG, oscc_no_temp_pool=True)
        self.num_classes = 16 if cfg.DATA.TASK == "keyframe_localization" else 2
        self.unsqueeze_dim = 1 if cfg.DATA.TASK == "keyframe_localization" else 2
        self.sequence_len = 32
        self.feature_dim = 256
        self.proj1 = nn.Linear(8192, self.feature_dim)
        self.proj2 = nn.Linear(8192, self.feature_dim)
        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        self.transformer = Transformer(dim=self.feature_dim, depth=3, heads=8, dim_head=128, mlp_dim=512)
        self.linear_head = nn.Sequential(nn.LayerNorm(self.feature_dim), nn.Linear(self.feature_dim, self.num_classes))

    def forward(self, x):
        x2 = x.copy()
        pnr_feat = self.pnr_model(x, middle=True)  # (bs, 16, 8192)
        oscc_feat = self.oscc_model(x2, middle=True)  # (bs, 16, 8192)
        feat = torch.cat((self.proj1(pnr_feat), self.proj2(oscc_feat)), dim=1) + self.pe # (bs, 32, 128)
        out = self.transformer(feat)
        out = out.mean(dim=1)
        out = self.linear_head(out)
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class TaskFusionMFTransformerDropout(TaskFusion):
    """mid fusion transformer"""
    def __init__(self, cfg):
        super(TaskFusionMFTransformerDropout, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, cfg.PRETRAIN.OSCC_CFG, oscc_no_temp_pool=True)
        self.num_classes = 16 if cfg.DATA.TASK == "keyframe_localization" else 2
        self.unsqueeze_dim = 1 if cfg.DATA.TASK == "keyframe_localization" else 2
        self.sequence_len = 32
        self.feature_dim = 256
        self.proj1 = nn.Linear(8192, self.feature_dim)
        self.proj2 = nn.Linear(8192, self.feature_dim)
        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.dp = nn.Dropout(cfg.MODEL.FEAT_DROPOUT_RATE)
        self.dpmode = cfg.MODEL.FEAT_DROPOUT_MODE
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8,
                                                     dropout=cfg.MODEL.TRANSFORMER_DROPOUT_RATE,
                                                     dim_feedforward=self.feature_dim * 2, batch_first=True),
            num_layers=3)
        self.linear_head = nn.Linear(self.feature_dim, self.num_classes)

    def forward(self, x):
        x2 = x.copy()
        pnr_feat = self.proj1(self.pnr_model(x, middle=True))  # (bs, 16, 8192)
        oscc_feat = self.proj2(self.oscc_model(x2, middle=True))  # (bs, 16, 8192)
        if self.dpmode > 0:
            pnr_feat = self.dp(pnr_feat)
        elif self.dpmode > 1:
            oscc_feat = self.dp(oscc_feat)
        feat = torch.cat((pnr_feat, oscc_feat), dim=1)
        feat = self.ln(feat) + self.pe # (bs, 32, 128)
        out = self.transformer(feat)
        out = out.mean(dim=1)
        out = self.linear_head(out)
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class TaskFusionLFLinear(TaskFusion):
    """naive late fusion"""
    def __init__(self, cfg):
        super(TaskFusionLFLinear, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, cfg.PRETRAIN.OSCC_CFG)
        self.num_classes = 16 if cfg.DATA.TASK == "keyframe_localization" else 2
        self.unsqueeze_dim = 1 if cfg.DATA.TASK == "keyframe_localization" else 2
        self.fc1 = nn.Linear(8192 * 2, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x2 = x.copy()
        pnr_feat = self.pnr_model(x, middle=True)  # (bs, 16, 8192)
        pnr_feat = pnr_feat.mean(dim=1)  # (bs, 8192)
        oscc_feat = self.oscc_model(x2, middle=True)  # (bs, 1, 8192)
        oscc_feat = oscc_feat.squeeze(1)  # (bs, 8192)

        feat = torch.cat((pnr_feat, oscc_feat), dim=1)  # (bs, 8192*2)
        out = self.act(self.fc1(feat))
        out = self.fc2(out)
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class Keyframe2State(TaskFusion):
    """transfer lr: from pnr -> oscc"""
    def __init__(self, cfg):
        super(Keyframe2State, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, None)
        self.num_classes = 2
        self.unsqueeze_dim = 2
        self.fc1 = nn.Linear(8192 * 2, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            pnr_feat = self.pnr_model(x, middle=True)  # (bs, 16, 8192)
            pnr_feat = pnr_feat.mean(dim=1)
        feat = torch.cat((pnr_feat, pnr_feat), dim=1)  # duplicate pnr frames for linear layer dimension consistency
        out = self.act(self.fc1(feat))
        out = self.fc2(out)
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class State2Keyframe(TaskFusion):
    """transfer lr: from oscc -> pnr"""
    def __init__(self, cfg):
        super(State2Keyframe, self).__init__(cfg, None, cfg.PRETRAIN.OSCC_CFG)
        self.num_classes = 16
        self.unsqueeze_dim = 1
        self.fc1 = nn.Linear(8192 * 2, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            oscc_feat = self.oscc_model(x, middle=True)  # (bs, 1, 8192)
            oscc_feat = oscc_feat.squeeze(1)  # (bs, 8192)
        feat = torch.cat((oscc_feat, oscc_feat), dim=1)  # duplicate pnr frames for linear layer dimension consistency
        out = self.act(self.fc1(feat))
        out = self.fc2(out)
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class FinetuneState(TaskFusion):
    """supervised lr: finetune oscc linear -> oscc"""
    def __init__(self, cfg):
        super(FinetuneState, self).__init__(cfg, None, cfg.PRETRAIN.OSCC_CFG)
        self.num_classes = 2
        self.unsqueeze_dim = 2
        self.fc1 = nn.Linear(8192 * 2, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            oscc_feat = self.oscc_model(x, middle=True)  # (bs, 1, 8192)
            oscc_feat = oscc_feat.squeeze(1)  # (bs, 8192)
        feat = torch.cat((oscc_feat, oscc_feat), dim=1)  # duplicate pnr frames for linear layer dimension consistency
        out = self.act(self.fc1(feat))
        out = self.fc2(out)
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class FinetuneKeyframe(TaskFusion):
    """supervised lr: finetune pnr linear -> pnr"""
    def __init__(self, cfg):
        super(FinetuneKeyframe, self).__init__(cfg, cfg.PRETRAIN.PNR_CFG, None)
        self.num_classes = 16
        self.unsqueeze_dim = 1
        self.fc1 = nn.Linear(8192 * 2, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            pnr_feat = self.pnr_model(x, middle=True)  # (bs, 16, 8192)
            pnr_feat = pnr_feat.mean(dim=1)
        feat = torch.cat((pnr_feat, pnr_feat), dim=1)  # duplicate pnr frames for linear layer dimension consistency
        out = self.act(self.fc1(feat))
        out = self.fc2(out)
        return out.unsqueeze(self.unsqueeze_dim)


@MODEL_REGISTRY.register()
class NoAuxTaskMFTransformer3TaskDropout(nn.Module):
    """mid fusion transformer"""
    def __init__(self, cfg):
        super(NoAuxTaskMFTransformer3TaskDropout, self).__init__()
        self.num_classes = 16 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.unsqueeze_dim = 1 if "keyframe_localization" in cfg.DATA.TASK else 2
        self.sequence_len = 48
        self.feature_dim = cfg.MODEL.TRANSLATION_INPUT_FEATURES
        self.num_layers = cfg.MODEL.TRANSLATION_LAYERS

        cfg_oscc = load_config_file(cfg.PRETRAIN.OSCC_CFG)
        cfg_oscc.MODEL.NO_TEMP_POOL = True
        self.oscc_model1 = StateChangeClsResNet(cfg_oscc)
        self.oscc_model2 = StateChangeClsResNet(cfg_oscc)
        self.oscc_model3 = StateChangeClsResNet(cfg_oscc)
        load_checkpoint(self.oscc_model1,
                        "/checkpoint/sherryxue/exp_fho/state_change_classification_adamw/lightning_logs/version_60686792/checkpoints/epoch=5-step=7698.ckpt")
        load_checkpoint(self.oscc_model2,
                        "/checkpoint/sherryxue/exp_fho/state_change_classification/lightning_logs/version_60544257/checkpoints/epoch54.ckpt")
        load_checkpoint(self.oscc_model3,
                        "/checkpoint/sherryxue/exp_fho/state_change_classification_fixloss/lightning_logs/version_60639193/checkpoints/epoch=0-step=1283.ckpt")

        self.oscc_model1.eval()
        self.oscc_model2.eval()
        self.oscc_model3.eval()
        freeze_params(self.oscc_model1)
        freeze_params(self.oscc_model2)
        freeze_params(self.oscc_model3)

        self.proj1 = nn.Linear(8192, self.feature_dim)
        self.proj2 = nn.Linear(8192, self.feature_dim)
        self.proj3 = nn.Linear(8192, self.feature_dim)

        self.pe = nn.Parameter(torch.randn(1, self.sequence_len, self.feature_dim), requires_grad=True)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.dp = nn.Dropout(cfg.MODEL.FEAT_DROPOUT_RATE)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8,
                                                     dropout=cfg.MODEL.TRANSFORMER_DROPOUT_RATE,
                                                     dim_feedforward=self.feature_dim * 2, batch_first=True),
            num_layers=self.num_layers)
        self.linear_head = nn.Sequential(self.ln, nn.Linear(self.feature_dim, self.num_classes))

    def forward(self, x):
        x2 = x.copy()
        x3 = x.copy()
        with torch.no_grad():
            x1 = self.oscc_model1(x, middle=True)  # (bs, 16, 8192)
            x2 = self.oscc_model2(x2, middle=True)
            x3 = self.oscc_model3(x3, middle=True)

        feat = torch.cat((self.dp(self.proj1(x1)), self.dp(self.proj2(x2)), self.dp(self.proj3(x3))), dim=1)  # (bs, seq_len, feat_dim)
        feat = self.ln(feat) + self.pe # (bs, 48, 128)
        out = self.transformer(feat)
        out = out.mean(dim=1)
        out = self.linear_head(out)
        return out.unsqueeze(self.unsqueeze_dim)