#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

"""Video models."""

import torch
import torch.nn as nn
import utils.pnr.weight_init_helper as init_helper
from models.pnr.batchnorm_helper import get_norm
from models.pnr.resnet import resnet50
from .build import MODEL_REGISTRY
from . import head_helper, resnet_helper, stem_helper

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slow_layer3":[
        [[1]],
        [[1]],
        [[3]],
        [[3]],
        [[3]],
    ],
    "slow_layer4":[
        [[1]],
        [[3]],
        [[3]],
        [[3]],
        [[3]],
    ],
    "slow_layer5":[
        [[3]],
        [[3]],
        [[3]],
        [[3]],
        [[3]],
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slow_layer3": [[1, 1, 1]],
    "slow_layer4": [[1, 1, 1]],
    "slow_layer5": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = is_detection_enabled(cfg)
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg, with_head=True):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if not with_head:
            return

        if self.enable_detection:
            head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES[0],
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES[0],
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        self.head_name = "head"
        self.add_module(self.head_name, head)

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)

        head = getattr(self, self.head_name)
        if self.enable_detection:
            x = head(x, bboxes)
        else:
            x = head(x)
        return x



@MODEL_REGISTRY.register()
class KeyframeLocalizationResNet(ResNet):
    def _construct_network(self, cfg):
        super()._construct_network(cfg, with_head=False)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]
        head_1 = head_helper.ResNetKeyframeLocalizationHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES[0],
            pool_size=[
            [
                1,
                cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
            ]
        ],
        dropout_rate=cfg.MODEL.DROPOUT_RATE,
        act_func=cfg.MODEL.KEYFRAME_DETECTION_ACT,
        )
        self.head_1_name = 'Keyframe_localisation_head'
        self.add_module(self.head_1_name, head_1)

    def forward(self, x, middle=False):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        head_1 = getattr(self, self.head_1_name)
        keyframe_output = head_1(x.copy(), middle)
        return keyframe_output


@MODEL_REGISTRY.register()
class StateChangeClsResNet(ResNet):
    def _construct_network(self, cfg):
        super()._construct_network(cfg, with_head=False)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]
        num_frames = cfg.DATA.CLIP_LEN_SEC * cfg.DATA.SAMPLING_FPS

        temp_pool_size = 1 if cfg.MODEL.NO_TEMP_POOL else num_frames // pool_size[0][0]
        head_2 = head_helper.ResNetKeyframeLocalizationHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_STATE_CLASSES[0],
            pool_size=[
                [
                temp_pool_size,
                cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ]
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.STATE_CHANGE_ACT,
        )
        self.head_2_name = 'State_detection_head'
        self.add_module(self.head_2_name, head_2)

    def forward(self, x, middle=False):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)

        head_2 = getattr(self, self.head_2_name)
        state_change_output = head_2(x.copy(), middle)
        return state_change_output


@MODEL_REGISTRY.register()
class DualHeadResNet(ResNet):
    def _construct_network(self, cfg):
        super()._construct_network(cfg, with_head=False)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]
        head_1 = head_helper.ResNetKeyframeLocalizationHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES[0],
            pool_size=[
            [
                1,
                cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
            ]
        ],
        dropout_rate=cfg.MODEL.DROPOUT_RATE,
        act_func=cfg.MODEL.KEYFRAME_DETECTION_ACT,
        )
        self.head_1_name = 'Keyframe_localisation_head'
        self.add_module(self.head_1_name, head_1)
        num_frames = cfg.DATA.CLIP_LEN_SEC * cfg.DATA.SAMPLING_FPS

        head_2 = head_helper.ResNetKeyframeLocalizationHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_STATE_CLASSES[0],
            pool_size=[
                [
                num_frames // pool_size[0][0],
                cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ]
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.STATE_CHANGE_ACT,
        )
        self.head_2_name = 'State_detection_head'
        self.add_module(self.head_2_name, head_2)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)

        head_1 = getattr(self, self.head_1_name)
        head_2 = getattr(self, self.head_2_name)

        keyframe_output = head_1(x.copy())
        state_change_output = head_2(x.copy())
        return keyframe_output, state_change_output


def is_detection_enabled(cfg):
    try:
        detection_enabled = cfg.DATA.TASK == "detection" or cfg.DATA.TASK=="short_term_anticipation"
    except Exception:
        # Temporary default config while still using old models without this config option
        detection_enabled = False

    return detection_enabled


@MODEL_REGISTRY.register()
class KeyframeCnnLSTM(nn.Module):
    def __init__(self, cfg):
        super(KeyframeCnnLSTM, self).__init__()
        self.hidden_size = 512
        self.num_layers = 1
        self.state = False
        self.backbone = resnet50(pretrained=True)
        # self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = None
        self.lstm = nn.LSTM(2048, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.regressor = nn.Linear(self.hidden_size * 2, 1)
        if self.state:
            self.state_classifier = nn.Linear(self.hidden_size * 2, 2)
        # self.backbone.forward = types.MethodType(forward_reimpl, self.backbone)

    def forward(self, x):
        # x: (b, c, seq_len, h, w)
        x = x[0]
        seq_len = x.shape[2]
        batch_size = x.shape[0]
        x = x.permute((0, 2, 1, 3, 4))
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.backbone(x)

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)  # (b, seq_len, hidden_size*2)
        out = self.regressor(x).squeeze(2)
        if self.state:
            state = self.state_classifier(x.mean(1))
            return torch.sigmoid(out), state
        return torch.sigmoid(out)


# from .video_model_transfer import TaskFusionMFTransformer, TaskFusionLFLinear, State2Keyframe, Keyframe2State, FinetuneKeyframe, FinetuneState
# MODEL_REGISTRY.register(TaskFusionMFTransformer)
# MODEL_REGISTRY.register(TaskFusionLFLinear)
# MODEL_REGISTRY.register(State2Keyframe)
# MODEL_REGISTRY.register(Keyframe2State)
# MODEL_REGISTRY.register(FinetuneKeyframe)
# MODEL_REGISTRY.register(FinetuneState)
#
# from .video_model_transfer_3task import TaskFusionMFTransformer3Task, Action2State, Action2Keyframe
# MODEL_REGISTRY.register(TaskFusionMFTransformer3Task)
# MODEL_REGISTRY.register(Action2State)
# MODEL_REGISTRY.register(Action2Keyframe)