#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

from models.lam.model import LAMBackbone
from models.ttm.model import TTMBackbone
from models.asd.talkNetModel import talkNetModel
from utils.utils import freeze_params, load_ckpt

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


class TaskPromptTransformer(nn.Module):
    def __init__(self, args, vocab):
        super(TaskPromptTransformer, self).__init__()
        self.args = args
        self.vocab = vocab
        self.n_tasks = 3
        self.dim = args.hidden_dim
        self.n_heads = args.num_heads
        self.num_layers = args.num_layers
        self.max_output_length = 500

        # transformer encoder and decoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.n_heads),
            num_layers=self.num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.dim, nhead=self.n_heads),
            num_layers=self.num_layers
        )
        self.ln = nn.LayerNorm(self.dim)
        self.task_embed = nn.Parameter(torch.randn(1, self.n_tasks, self.dim), requires_grad=True)
        self.pos_embed = PositionalEncoding(self.dim, dropout=0.1)
        self.embedding = nn.Embedding(len(self.vocab), self.dim)
        self.proj_lam = nn.Linear(256, self.dim)
        self.proj_ttm = nn.Linear(256, self.dim)
        self.proj_asd = nn.Linear(256, self.dim)
        self.fc = nn.Linear(self.dim, len(self.vocab))
        # sequence length is always 2 for now
        self.seq_len = 2
        self.y_mask = self.get_tgt_mask(self.seq_len)

        self._init_parameters()

        # load pretrained models
        self.lam_model = LAMBackbone(args.lam_checkpoint)
        self.ttm_model = TTMBackbone(args.ttm_checkpoint)
        self.asd_model = talkNetModel()
        load_ckpt(self.asd_model, args.asd_checkpoint, load_asd=True)

        freeze_params(self.lam_model)
        freeze_params(self.ttm_model)
        freeze_params(self.asd_model)

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_prepare(self, x, task_id):
        x = self.ln(x) + self.task_embed[:, task_id, :]
        x = x.permute(1, 0, 2)  #(#frame, bs, 256)
        x = self.pos_embed(x)
        return x

    def encode(self, video, task, audio=None):
        if task == 'lam':
            with torch.no_grad():
                lam_feat = self.lam_model(video, middle=True)  # (bs, #frame-7, dim-256)
            x = self.encode_prepare(self.proj_lam(lam_feat), task_id=0)  # (7, bs, 256)
        elif task == 'ttm':
            with torch.no_grad():
                ttm_feat = self.ttm_model(video, audio, middle=True)  #(bs, #frame, 256)
            x = self.encode_prepare(self.proj_ttm(ttm_feat), task_id=1)
        else:
            with torch.no_grad():
                N, D, H, W = video.shape
                audioEmbed = self.asd_model.forward_audio_frontend(audio)
                visualEmbed = self.asd_model.forward_visual_frontend(video)
                audioEmbed, visualEmbed = self.asd_model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV = self.asd_model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                asd_feat = outsAV.view(N, D, -1)  #(bs, #frame, 256)
            x = self.encode_prepare(self.proj_asd(asd_feat), task_id=2)
        return x


    def forward(self, video, target, task, audio=None):
        assert task in ['lam', 'ttm', 'asd']
        x = self.encode(video, task, audio)
        encoded_x = self.transformer_encoder(x)
        if task == 'asd':
            encoded_x = encoded_x.permute(1, 0, 2)
            n, d, _ = encoded_x.shape
            encoded_x = encoded_x.reshape(1, n*d, -1)

        output = self.decode(target, encoded_x)
        return output.permute(1, 2, 0)   # (bs, vocab_size, seq_y)


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

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        return mask

    def predict(self, video, task, audio=None):
        assert task in ['lam', 'ttm', 'asd']
        batch_size = video.shape[0] * video.shape[1] if task == 'asd' else video.shape[0]
        x = self.encode(video, task, audio)
        encoded_x = self.transformer_encoder(x)
        if task == 'asd':
            encoded_x = encoded_x.permute(1, 0, 2)
            n, d, _ = encoded_x.shape
            encoded_x = encoded_x.reshape(1, n * d, -1)

        y = torch.ones((batch_size, 1)) * self.vocab[task]
        y = y.type_as(video).long()
        output = self.decode(y, encoded_x)
        return output[0, :, -2:]  # 0,1 correspond to last two elements in vocab


class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CustomDecoderLayer, self).__init__(d_model, nhead, dropout=dropout)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)[0]
        return self.dropout2(x)

class TaskTranslationPromptTransformer(nn.Module):
    def __init__(self, args, vocab):
        super(TaskTranslationPromptTransformer, self).__init__()
        self.args = args
        self.vocab = vocab
        self.n_tasks = 3
        self.dim = args.hidden_dim
        self.n_heads = args.num_heads
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
        self.ln = nn.LayerNorm(self.dim)
        self.task_embed = nn.Parameter(torch.randn(1, self.n_tasks, self.dim), requires_grad=True)
        self.pos_embed = PositionalEncoding(self.dim, dropout=0.1)
        self.embedding = nn.Embedding(len(self.vocab), self.dim)
        self.proj_lam = nn.Linear(256, self.dim)
        self.proj_ttm = nn.Linear(256, self.dim)
        self.proj_asd = nn.Linear(256, self.dim)
        self.fc = nn.Linear(self.dim, len(self.vocab))
        # sequence length is always 2 for now
        self.seq_len = 2
        self.y_mask = self.get_tgt_mask(self.seq_len)

        self._init_parameters()

        # load pretrained models
        self.lam_model = LAMBackbone(args.lam_checkpoint)
        self.ttm_model = TTMBackbone(args.ttm_checkpoint)
        self.asd_model = talkNetModel()
        load_ckpt(self.asd_model, args.asd_checkpoint, load_asd=True)

        freeze_params(self.lam_model)
        freeze_params(self.ttm_model)
        freeze_params(self.asd_model)

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_prepare(self, x, task_id):
        x = self.ln(x) + self.task_embed[:, task_id, :]
        x = x.permute(1, 0, 2)  #(#frame, bs, 256)
        x = self.pos_embed(x)
        return x

    def encode(self, video, video_asd, audio, audio_asd, task):
        if task == 'lam':
            with torch.no_grad():
                lam_feat = self.lam_model(video, middle=True)  # (bs, #frame-7, dim-256)
            x = self.encode_prepare(self.proj_lam(lam_feat), task_id=0)  # (7, bs, 256)
        else:
            with torch.no_grad():
                lam_feat = self.lam_model(video, middle=True)  # (bs, #frame-7, dim-256)
                ttm_feat = self.ttm_model(video, audio, middle=True)  #(bs, #frame, 256)
                N, D, H, W = video_asd.shape
                audioEmbed = self.asd_model.forward_audio_frontend(audio_asd)
                visualEmbed = self.asd_model.forward_visual_frontend(video_asd)
                audioEmbed, visualEmbed = self.asd_model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV = self.asd_model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                asd_feat = outsAV.view(N, D, -1)  # (bs, #frame, 256)
            x1 = self.encode_prepare(self.proj_lam(lam_feat), task_id=0)
            x2 = self.encode_prepare(self.proj_ttm(ttm_feat), task_id=1)
            x3 = self.encode_prepare(self.proj_asd(asd_feat), task_id=2)
            x = torch.cat((x1, x2, x3), dim=0)  # (3T, bs, dim)

        encoded_x = self.transformer_encoder(x)
        if task == 'asd':
            encoded_x = encoded_x.permute(1, 0, 2)  # (bs, 3T, dim)
            T = int(encoded_x.shape[1] / 3)
            tmp1 = encoded_x[:, 0: T, :].reshape(-1, self.dim)
            tmp2 = encoded_x[:, T: 2*T, :].reshape(-1, self.dim)
            tmp3 = encoded_x[:, 2*T: 3*T, :].reshape(-1, self.dim)
            encoded_x = torch.stack((tmp1, tmp2, tmp3), dim=0)
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

    def forward(self, video, video_asd, audio, audio_asd, target, task):
        assert task in ['lam', 'ttm', 'asd']
        encoded_x = self.encode(video, video_asd, audio, audio_asd, task)
        output = self.decode(target, encoded_x)
        return output.permute(1, 2, 0)   # (bs, vocab_size, seq_y)


    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        return mask

    def predict(self, video, video_asd, audio, audio_asd, task):
        assert task in ['lam', 'ttm', 'asd']
        batch_size = video.shape[0] * video.shape[1] if task == 'asd' else video.shape[0]
        encoded_x = self.encode(video, video_asd, audio, audio_asd, task)
        y = torch.ones((batch_size, 1)) * self.vocab[task]
        y = y.type_as(video).long()
        output = self.decode(y, encoded_x)
        return output[0, :, -2:]  # 0,1 correspond to last two elements in vocab
