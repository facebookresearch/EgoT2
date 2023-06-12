#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from .build import MODEL_REGISTRY
from .resnet import resnet18
from .resse import ResNetSE

logger = logging.getLogger(__name__)


class TTMBackbone(nn.Module):
    def __init__(self, checkpoint):
        super(TTMBackbone, self).__init__()
        self.img_feature_dim = 256
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.video_encoder = resnet18(pretrained=False)
        self.video_encoder.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2,
                            batch_first=True)
        self.audio_encoder = ResNetSE()
        self.load_checkpoint(checkpoint)

    def forward(self, video, audio, middle=False):
        N, D, C, H, W = video.shape
        video_out = self.video_encoder(video.view(N * D, C, H, W))
        video_out = video_out.view(N, D, self.img_feature_dim)
        if middle:
            return video_out
        lstm_out, _ = self.lstm(video_out)
        lstm_out = lstm_out[:, -1, :]

        audio_out = self.audio_encoder(audio)
        # output = self.last_layer1(torch.cat((lstm_out, audio_out), dim=1))
        # output = self.last_layer2(output)
        return lstm_out, audio_out

    def load_checkpoint(self, checkpoint):
        print(f'loading checkpoint {checkpoint}')
        state = torch.load(checkpoint)
        if 'module' in list(state["state_dict"].keys())[0]:
            state_dict = {k[7:]: v for k, v in state["state_dict"].items()}
        else:
            state_dict = state["state_dict"]
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print('missing', missing_keys)
        print('unexpected', unexpected_keys)


@MODEL_REGISTRY.register()
class BaselineLSTM(nn.Module):
    def __init__(self, args):
        super(BaselineLSTM, self).__init__()
        self.args = args
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.video_encoder = resnet18(pretrained=False)

        self.video_encoder.fc2 = nn.Linear(1000, self.img_feature_dim)

        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2,
                            batch_first=True)

        self.audio_encoder = ResNetSE()

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer1 = nn.Linear(4 * self.img_feature_dim, 128)

        self.last_layer2 = nn.Linear(128, 2)

        for param in self.parameters():
            param.requires_grad = True

        self._init_parameters()

        self.load_checkpoint()

    def forward(self, video, audio):
        N, D, C, H, W = video.shape

        video_out = self.video_encoder(video.view(N * D, C, H, W))
        video_out = video_out.view(N, D, self.img_feature_dim)

        lstm_out, _ = self.lstm(video_out)
        lstm_out = lstm_out[:, -1, :]

        audio_out = self.audio_encoder(audio)

        output = self.last_layer1(torch.cat((lstm_out, audio_out), dim=1))

        output = self.last_layer2(output)

        return output

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f'loading checkpoint {self.args.checkpoint}')
                state = torch.load(self.args.checkpoint, map_location=f'cuda:{self.args.rank}')
                if 'module' in list(state["state_dict"].keys())[0]:
                    state_dict = {k[7:]: v for k, v in state["state_dict"].items()}
                else:
                    state_dict = state["state_dict"]
                self.load_state_dict(state_dict)

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)