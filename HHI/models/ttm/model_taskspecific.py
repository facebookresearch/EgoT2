#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from models.lam.model import LAMBackbone
from models.ttm.model import TTMBackbone
from models.asd.talkNetModel import talkNetModel
from utils.utils import load_ckpt, freeze_params
from .build import MODEL_REGISTRY
import math


class TaskFusion3Task(nn.Module):
    def __init__(self, lam_ckpt=None, ttm_ckpt=None, asd_ckpt=None, nofreeze=False):
        super(TaskFusion3Task, self).__init__()
        if lam_ckpt:
            self.lam_model = LAMBackbone(lam_ckpt)
            freeze_params(self.lam_model)
        if ttm_ckpt:
            self.ttm_model = TTMBackbone(ttm_ckpt)
        if asd_ckpt:
            self.asd_model = talkNetModel()
            load_ckpt(self.asd_model, asd_ckpt, load_asd=True)
            freeze_params(self.asd_model)

        if not nofreeze:
            print('Freezing task-specific models')
            freeze_params(self.ttm_model)

    def forward(self, video, video_asd, audio, audio_asd):
        raise NotImplementedError


@MODEL_REGISTRY.register()
class FinetuneTTM(TaskFusion3Task):
    """Fine-tuning Baseline"""
    def __init__(self, args):
        super(FinetuneTTM, self).__init__(None, args.ttm_checkpoint, None)
        self.fc1 = nn.Linear(256, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim2)
        self.fc3 = nn.Linear(args.hidden_dim2, 2)
        self.act = nn.ReLU()

    def forward(self, video, video_asd, audio, audio_asd):
        with torch.no_grad():
            ttm_out = self.ttm_model(video, audio, middle=True).mean(dim=1)  # (bs, 256)
        out = self.act(self.fc1(ttm_out))
        out = self.act(self.fc2(out))
        out = self.fc3(out)
        return out


@MODEL_REGISTRY.register()
class LAM2TTM(TaskFusion3Task):
    """Transfer Learning Baseline"""
    def __init__(self, args):
        super(LAM2TTM, self).__init__(args.lam_checkpoint, None, None)
        self.fc1 = nn.Linear(256, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim2)
        self.fc3 = nn.Linear(args.hidden_dim2, 2)
        self.act = nn.ReLU()

    def forward(self, video, video_asd, audio, audio_asd):
        with torch.no_grad():
            lam_out = self.lam_model(video, middle=True).mean(dim=1)  # (bs, 256)
        out = self.act(self.fc1(lam_out))
        out = self.act(self.fc2(out))
        out = self.fc3(out)
        return out


@MODEL_REGISTRY.register()
class ASD2TTM(TaskFusion3Task):
    """Transfer Learning Baseline"""
    def __init__(self, args):
        super(ASD2TTM, self).__init__(None, None, args.asd_checkpoint)
        self.fc1 = nn.Linear(256, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim2)
        self.fc3 = nn.Linear(args.hidden_dim2, 2)
        self.act = nn.ReLU()

    def forward(self, video, video_asd, audio, audio_asd):
        with torch.no_grad():
            N, D, H, W = video_asd.shape
            audioEmbed = self.asd_model.forward_audio_frontend(audio_asd)
            visualEmbed = self.asd_model.forward_visual_frontend(video_asd)
            audioEmbed, visualEmbed = self.asd_model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV = self.asd_model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            asd_out = outsAV.view(N, D, -1).mean(dim=1)  # (bs, 256)
        out = self.act(self.fc1(asd_out))
        out = self.act(self.fc2(out))
        out = self.fc3(out)
        return out


@MODEL_REGISTRY.register()
class TaskFusionLFLinear3Task(TaskFusion3Task):
    """Late Fusion Baseline"""
    def __init__(self, args):
        super(TaskFusionLFLinear3Task, self).__init__(args.lam_checkpoint, args.ttm_checkpoint, args.asd_checkpoint)
        self.dim = args.hidden_dim
        self.dim2 = args.hidden_dim2
        self.proj_lam = nn.Linear(256, self.dim)
        self.proj_ttm = nn.Linear(256, self.dim)
        self.proj_asd = nn.Linear(256, self.dim)
        self.ln = nn.LayerNorm(self.dim * 3)
        self.fc1 = nn.Linear(self.dim * 3, self.dim2)
        self.fc2 = nn.Linear(self.dim2, 2)
        self.act = nn.ReLU()

    def forward(self, video, video_asd, audio, audio_asd):
        with torch.no_grad():
            N, D, H, W = video_asd.shape
            audioEmbed = self.asd_model.forward_audio_frontend(audio_asd)
            visualEmbed = self.asd_model.forward_visual_frontend(video_asd)
            audioEmbed, visualEmbed = self.asd_model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV = self.asd_model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            asd_out = outsAV.view(N, D, -1).mean(dim=1)  # (bs, 256)
            lam_out = self.lam_model(video, middle=True).mean(dim=1)  # (bs, 256)
            ttm_out = self.ttm_model(video, audio, middle=True).mean(dim=1)  # (bs, 256)
        feat = torch.cat((self.proj_ttm(ttm_out), self.proj_lam(lam_out), self.proj_asd(asd_out)), dim=1)
        out = self.fc1(self.ln(feat))
        out = self.fc2(self.act(out))
        return out


class PositionalEncoding(nn.Module):
    """
    Classic Attention-is-all-you-need positional encoding.
    From PyTorch docs.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1000):
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
class TaskFusionMFTransformer2Task(TaskFusion3Task):
    """Task Translation for 2 tasks: LAM and TTM"""
    def __init__(self, args):
        super(TaskFusionMFTransformer2Task, self).__init__(args.lam_checkpoint, args.ttm_checkpoint, None, args.nofreeze)
        self.n_tasks = 2
        self.dim = args.hidden_dim
        self.n_heads = args.num_heads
        self.dp_rate = args.dropout
        self.num_layers = args.num_layers
        self.proj_lam = nn.Linear(256, self.dim)
        self.proj_ttm = nn.Linear(256, self.dim)
        self.task_embed = nn.Parameter(torch.randn(1, self.n_tasks, self.dim), requires_grad=True)
        self.pos_embed = PositionalEncoding(self.dim, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.n_heads, dropout=self.dp_rate),
            num_layers=self.num_layers
        )
        self.ln = nn.LayerNorm(self.dim)
        self.linear_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 2)
        )

    def encode_prepare(self, x, task_id):
        x = self.ln(x) + self.task_embed[:, task_id, :]
        x = x.permute(1, 0, 2)  #(#frame, bs, 256)
        x = self.pos_embed(x)
        return x

    def forward(self, video, audio):
        lam_out = self.lam_model(video, middle=True)  # (bs, D2, 256)
        ttm_out = self.ttm_model(video, audio, middle=True)

        x1 = self.encode_prepare(self.proj_ttm(ttm_out), 0)
        x2 = self.encode_prepare(self.proj_lam(lam_out), 1)
        feat = torch.cat((x1, x2), dim=0)  # (D, bs, 256)
        output = self.transformer_encoder(feat)
        output = output.mean(dim=0)  # other way: use cls token
        output = self.linear_head(output)
        return output


@MODEL_REGISTRY.register()
class TaskFusionMFTransformer3Task(TaskFusion3Task):
    """Task Translation for 3 HHI tasks: LAM, TTM, ASD"""
    def __init__(self, args):
        super(TaskFusionMFTransformer3Task, self).__init__(args.lam_checkpoint, args.ttm_checkpoint, args.asd_checkpoint, args.nofreeze)
        self.n_tasks = 3
        self.dim = args.hidden_dim
        self.n_heads = args.num_heads
        self.dp_rate = args.dropout
        self.num_layers = args.num_layers
        self.proj_lam = nn.Linear(256, self.dim)
        self.proj_ttm = nn.Linear(256, self.dim)
        self.proj_asd = nn.Linear(256, self.dim)
        self.task_embed = nn.Parameter(torch.randn(1, self.n_tasks, self.dim), requires_grad=True)
        self.pos_embed = PositionalEncoding(self.dim, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.n_heads, dropout=self.dp_rate),
            num_layers=self.num_layers
        )
        self.ln = nn.LayerNorm(self.dim)
        self.linear_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 2)
        )

    def encode_prepare(self, x, task_id):
        x = self.ln(x) + self.task_embed[:, task_id, :]
        x = x.permute(1, 0, 2)  # (#frame, bs, 256)
        x = self.pos_embed(x)
        return x

    def forward(self, video, video_asd, audio, audio_asd):
        N, D, H, W = video_asd.shape
        audioEmbed = self.asd_model.forward_audio_frontend(audio_asd)
        visualEmbed = self.asd_model.forward_visual_frontend(video_asd)
        audioEmbed, visualEmbed = self.asd_model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.asd_model.forward_audio_visual_backend(audioEmbed, visualEmbed)
        asd_out = outsAV.view(N, D, -1)  # (bs, D1, 256)
        lam_out = self.lam_model(video, middle=True)  # (bs, D2, 256)
        ttm_out = self.ttm_model(video, audio, middle=True)

        x1 = self.encode_prepare(self.proj_ttm(ttm_out), 0)
        x2 = self.encode_prepare(self.proj_lam(lam_out), 1)
        x3 = self.encode_prepare(self.proj_asd(asd_out), 2)
        feat = torch.cat((x1, x2, x3), dim=0)  # (D, bs, 256)
        output = self.transformer_encoder(feat)
        output = output.mean(dim=0)  # other way: use cls token
        output = self.linear_head(output)
        return output