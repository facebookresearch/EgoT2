#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math
from .build import MODEL_REGISTRY
from models.lam.model import LAMBackbone
from models.ttm.model import TTMBackbone
from models.asd.talkNetModel import talkNetModel
from utils.utils import load_ckpt, freeze_params


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


class TaskFusion3Task(nn.Module):
    def __init__(self, lam_ckpt=None, ttm_ckpt=None, asd_ckpt=None):
        super(TaskFusion3Task, self).__init__()
        if lam_ckpt:
            self.lam_model = LAMBackbone(lam_ckpt)
            freeze_params(self.lam_model)
        if ttm_ckpt:
            self.ttm_model = TTMBackbone(ttm_ckpt)
            freeze_params(self.ttm_model)
        if asd_ckpt:
            self.asd_model = talkNetModel()
            load_ckpt(self.asd_model, asd_ckpt, load_asd=True)
            freeze_params(self.asd_model)

    def forward(self, video, video_asd, audio, audio_asd):
        raise NotImplementedError


@MODEL_REGISTRY.register()
class FinetuneASD(TaskFusion3Task):
    def __init__(self, args):
        super(FinetuneASD, self).__init__(None, None, args.asd_checkpoint)
        self.fc1 = nn.Linear(256, args.hidden_dim)
        self.act = nn.ReLU()
        self.output_dim = args.hidden_dim

    def forward(self, video, video_asd, audio, audio_asd):
        with torch.no_grad():
            audioEmbed = self.asd_model.forward_audio_frontend(audio_asd)
            visualEmbed = self.asd_model.forward_visual_frontend(video_asd)
            audioEmbed, visualEmbed = self.asd_model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV = self.asd_model.forward_audio_visual_backend(audioEmbed, visualEmbed)  #(N*D, 256)
        out = self.act(self.fc1(outsAV))
        return out


@MODEL_REGISTRY.register()
class LAM2ASD(TaskFusion3Task):
    def __init__(self, args):
        super(LAM2ASD, self).__init__(args.lam_checkpoint, None, None)
        self.fc1 = nn.Linear(256, args.hidden_dim)
        self.act = nn.ReLU()
        self.output_dim = args.hidden_dim

    def forward(self, video, video_asd, audio, audio_asd):
        with torch.no_grad():
            lam_out = self.lam_model(video, middle=True)
            N, D, _ = lam_out.shape
        out = self.act(self.fc1(lam_out))
        return out.reshape(N*D, -1)


@MODEL_REGISTRY.register()
class TTM2ASD(TaskFusion3Task):
    def __init__(self, args):
        super(TTM2ASD, self).__init__(None, args.ttm_checkpoint, None)
        self.fc1 = nn.Linear(256, args.hidden_dim)
        self.act = nn.ReLU()
        self.output_dim = args.hidden_dim

    def forward(self, video, video_asd, audio, audio_asd):
        with torch.no_grad():
            ttm_out = self.ttm_model(video, audio, middle=True)
            N, D, _ = ttm_out.shape
        out = self.act(self.fc1(ttm_out))
        return out.reshape(N*D, -1)


@MODEL_REGISTRY.register()
class TaskFusionMFTransformer3Task(TaskFusion3Task):
    def __init__(self, args):
        super(TaskFusionMFTransformer3Task, self).__init__(args.lam_checkpoint, args.ttm_checkpoint, args.asd_checkpoint)
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
        self.output_dim = self.dim

    def encode_prepare(self, x, task_id):
        x = self.ln(x) + self.task_embed[:, task_id, :]
        x = x.permute(1, 0, 2)  #(#frame, bs, 256)
        x = self.pos_embed(x)
        return x

    def forward(self, video, video_asd, audio, audio_asd):
        with torch.no_grad():
            N, D, H, W = video_asd.shape
            # video_ttm = video.view(N * D, *video.shape[2:])
            audioEmbed = self.asd_model.forward_audio_frontend(audio_asd)
            visualEmbed = self.asd_model.forward_visual_frontend(video_asd)
            audioEmbed, visualEmbed = self.asd_model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV = self.asd_model.forward_audio_visual_backend(audioEmbed, visualEmbed)  #(N*D, 256)
            asd_out = outsAV.view(N, D, -1)
            lam_out = self.lam_model(video, middle=True)
            ttm_out = self.ttm_model(video, audio, middle=True)

        x1 = self.encode_prepare(self.proj_ttm(ttm_out), 0)
        x2 = self.encode_prepare(self.proj_lam(lam_out), 1)
        x3 = self.encode_prepare(self.proj_asd(asd_out), 2)
        feat = torch.cat((x3, x1, x2), dim=0)  # (D, bs, 256)
        output = self.transformer_encoder(feat)
        output = output.permute(1, 0, 2)
        output = output[:, 0:D, :].reshape(N*D, -1)
        return output