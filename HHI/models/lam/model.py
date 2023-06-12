#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch, os, math, logging
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.nn.init import normal, constant
from .resnet import resnet18
from .build import MODEL_REGISTRY
logger = logging.getLogger(__name__)


class LAMBackbone(nn.Module):
    def __init__(self, checkpoint):
        super(LAMBackbone, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet18(pretrained=False)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2,
                            batch_first=True)
        self.load_checkpoint(checkpoint)

    def forward(self, input, middle=False):
        N, D, C, H, W = input.shape
        base_out = self.base_model(input.view(N * D, C, H, W))
        base_out = base_out.view(N, D, self.img_feature_dim)
        if middle:
            return base_out
        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:, D // 2, :]  # use the middle frame for now, are there other ways?
        return lstm_out

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
        self.base_model = resnet18(pretrained=False)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2,
                            batch_first=True)
        self.last_layer1 = nn.Linear(2 * self.img_feature_dim, 128)
        self.last_layer2 = nn.Linear(128, 2)

        for param in self.parameters():
            param.requires_grad = True

        self._init_parameters()

        self.load_checkpoint()

    def forward(self, input):
        N, D, C, H, W = input.shape
        base_out = self.base_model(input.view(N * D, C, H, W))
        base_out = base_out.view(N, D, self.img_feature_dim)
        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:, 3, :]
        output = self.last_layer1(lstm_out)
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


@MODEL_REGISTRY.register()
class GazeLSTM(nn.Module):
    def __init__(self, args):
        super(GazeLSTM, self).__init__()
        self.args = args
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet18(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2,
                            batch_first=True)
        # The linear layer that maps the LSTM with the 2 outputs
        self.last_layer1 = nn.Linear(2 * self.img_feature_dim, 128)
        self.last_layer2 = nn.Linear(128, 2)
        self.load_checkpoint()

    def forward(self, input):
        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))
        base_out = base_out.view(input.size(0), 7, self.img_feature_dim)
        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:, 3, :]
        output = self.last_layer1(lstm_out)
        output = self.last_layer2(output).view(-1, 2)
        return output

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f'loading checkpoint {self.args.checkpoint}')
                map_loc = f'cuda:{self.args.rank}' if torch.cuda.is_available() else 'cpu'
                state = torch.load(self.args.checkpoint, map_location=map_loc)
                if 'module' in list(state["state_dict"].keys())[0]:
                    state_dict = {k[7:]: v for k, v in state["state_dict"].items()}
                else:
                    state_dict = state["state_dict"]
                if 'gaze360' in self.args.checkpoint:
                    state_dict.pop('last_layer.weight')
                    state_dict.pop('last_layer.bias')
                self.load_state_dict(state_dict, strict=self.args.eval)