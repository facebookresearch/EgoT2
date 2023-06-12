#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import itertools
import os

import torch
import torch.utils.data
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import (
    Compose,
    Lambda,
)
from pytorchvideo.data.clip_sampling import ClipSampler, ClipInfo
from typing import Dict, Any

from .build import DATASET_REGISTRY
from . import ptv_dataset_helper
from utils.lta import logging, video_transformer
from dataset.lta.ptv_dataset_helper import LabeledVideoDataset, UntrimmedClipSampler

logger = logging.get_logger(__name__)


def make_transform_unlabeled(mode, cfg):
        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(lambda x: x/255.0),
                            Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
                        ]
                        + video_transformer.random_scale_crop_flip(mode, cfg)
                        + [video_transformer.uniform_temporal_subsample_repeated(cfg)]
                    ),
                ),
                Lambda(
                    lambda x: (
                        x["video"],  # only return video
                        {"video_name": x["video_name"], "start_sec": x["clip_start_sec"], "end_sec": x["clip_end_sec"]},
                    )
                ),
            ]
        )

def make_transform(mode: str, cfg):
    return Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                        Lambda(lambda x: x / 255.0),
                        Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
                    ]
                    + video_transformer.random_scale_crop_flip(mode, cfg)
                    + [video_transformer.uniform_temporal_subsample_repeated(cfg)]
                ),
            ),
            Lambda(
                lambda x: (
                    x["video"],
                    torch.tensor([x["verb_label"], x["noun_label"]]),
                    str(x["video_name"]) + "_" + str(x["video_index"]),
                    {"video_name": x["video_name"], "start_sec": x["clip_start_sec"], "end_sec": x["clip_end_sec"]},
                )
            ),
        ]
    )

class CenterClipVideoSampler(ClipSampler):
    """
    Samples just a single clip from the center of the video (use for testing)
    """

    def __init__(
        self, clip_duration: float
    ) -> None:
        super().__init__(clip_duration)

    def __call__(
        self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]
    ) -> ClipInfo:

        clip_start_sec = video_duration / 2 - self._clip_duration / 2
        # print(annotation, clip_start_sec, clip_start_sec+self._clip_duration)
        return ClipInfo(
            clip_start_sec,
            clip_start_sec + self._clip_duration,
            0,
            0,
            True,
        )

@DATASET_REGISTRY.register()
class Ego4dRecognition(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Ego4d ".format(mode)

        sampler = RandomSampler
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler

        clip_sampler_type = "uniform" if mode == "test" or "val" else "random"
        clip_duration = (
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        ) / self.cfg.DATA.TARGET_FPS
        # clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)
        if mode == "test" or "val":
            print('Center Clip Sampler')
            clip_sampler = CenterClipVideoSampler(clip_duration)
        else:
            clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)
        print(f"Clip sampler type {clip_sampler_type}")

        mode_ = 'test_unannotated' if mode=='test' else mode
        data_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'fho_lta_{mode_}.json')
        self.ann_file = data_path
        self.dataset = ptv_dataset_helper.clip_recognition_dataset(
            data_path=data_path,
            clip_sampler=clip_sampler,
            video_sampler=sampler,
            decode_audio=False,
            transform=make_transform(mode, cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
        )
        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 2)
        )

    @property
    def sampler(self):
        return self.dataset.video_sampler

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos


from utils.multitask.build_vocab import map_label_to_action
@DATASET_REGISTRY.register()
class Ego4dRecognitionSequenceLabel(Ego4dRecognition):
    def __init__(self, cfg, vocab, mode):
        super(Ego4dRecognitionSequenceLabel, self).__init__(cfg, mode)
        self.vocab = vocab
        self.verb_dict, self.noun_dict = map_label_to_action()
        self.action_task_idx = vocab['action']
        self.eos_idx = vocab['</s>']

    def __getitem__(self, index):
        inputs, labels, a, b = next(self._dataset_iter)
        verb = self.verb_dict[labels[0].item()]
        noun = self.noun_dict[labels[1].item()]
        target_seq = [self.action_task_idx, self.vocab[verb], self.vocab[noun], self.eos_idx]
        # print(a, labels, verb, noun, target_seq)
        return inputs, torch.LongTensor(target_seq), a, b, labels


@DATASET_REGISTRY.register()
class Ego4dRecognitionSeparateSequenceLabel(Ego4dRecognition):
    def __init__(self, cfg, vocab, mode):
        super(Ego4dRecognitionSeparateSequenceLabel, self).__init__(cfg, mode)
        self.vocab = vocab
        self.verb_dict, self.noun_dict = map_label_to_action()
        self.action_verb_idx = vocab['action_verb']
        self.action_noun_idx = vocab['action_noun']
        self.eos_idx = vocab['</s>']

    def __getitem__(self, index):
        inputs, labels, a, b = next(self._dataset_iter)
        verb = self.verb_dict[labels[0].item()]
        noun = self.noun_dict[labels[1].item()]
        target_seq_verb = [self.action_verb_idx, self.vocab[verb], self.eos_idx]
        target_seq_noun = [self.action_noun_idx, self.vocab[noun], self.eos_idx]
        return inputs, torch.LongTensor(target_seq_verb), torch.LongTensor(target_seq_noun), a, b, labels


@DATASET_REGISTRY.register()
class Ego4dLongTermAnticipation(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Ego4d ".format(mode)

        sampler = RandomSampler
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler

        clip_sampler_type = "uniform" if mode == "test" or "val" else "random"
        clip_duration = (
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        ) / self.cfg.DATA.TARGET_FPS
        print(f'clip sampler type {clip_sampler_type} clip duration {clip_duration}')

        if mode == "test" or "val":
            print('Center Clip Sampler')
            clip_sampler = CenterClipVideoSampler(clip_duration)
        else:
            clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)
        # mode_ = "val" if mode == 'test' else mode
        mode_ = 'test_unannotated' if mode =='test' else mode

        data_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'fho_lta_{mode_}.json')
        self.ann_file = data_path
        self.dataset, self.clip_annotations = ptv_dataset_helper.clip_forecasting_dataset(
            data_path=data_path,
            clip_sampler=clip_sampler,
            num_input_actions=self.cfg.FORECASTING.NUM_INPUT_CLIPS,
            num_future_actions=self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT,
            video_sampler=sampler,
            decode_audio=False,
            transform=self._make_transform(mode, cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
        )
        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 2)
        )

    @property
    def sampler(self):
        return self.dataset.video_sampler

    def _make_transform(self, mode: str, cfg):
        class ReduceExpandInputClips:
            def __init__(self, transform):
                self.transform = transform

            def __call__(self, x):
                if x.dim() == 4:
                    x = x.unsqueeze(0)  # Handle num_clips=1

                n, c, t, h, w = x.shape
                x = x.transpose(0, 1)
                x = x.reshape(c, n * t, h, w)
                x = self.transform(x)

                if isinstance(x, list):
                    for i in range(len(x)):
                        c, _, h, w = x[i].shape
                        x[i] = x[i].reshape(c, n, -1, h, w)
                        x[i] = x[i].transpose(1, 0)
                else:
                    c, _, h, w = x.shape
                    x = x.reshape(c, n, t, h, w)
                    x = x.transpose(1, 0)

                return x

        def extract_forecast_labels(x):
            clips = x["forecast_clips"]
            nouns = torch.tensor([y["noun_label"] for y in clips])
            verbs = torch.tensor([y["verb_label"] for y in clips])
            labels = torch.stack([verbs, nouns], dim=-1)
            return labels

        def extract_observed_labels(x):
            clips = x["input_clips"]
            nouns = torch.tensor([y["noun_label"] for y in clips])
            verbs = torch.tensor([y["verb_label"] for y in clips])
            labels = torch.stack([verbs, nouns], dim=-1)
            return labels

        # last visible annotated clip: (clip_uid + action_idx)
        def extract_clip_id(x):
            last_clip = x['input_clips'][-1]
            return f'{last_clip["clip_uid"]}_{last_clip["action_idx"]}'

        def extract_forecast_times(x):
            clips = x["forecast_clips"]
            start_end = [(y["clip_start_sec"], y["clip_end_sec"]) for y in clips]
            return {"label_clip_times": start_end}

        def extract_clip_info(x):
            clips = x['input_clips']
            clip_name = [clip['clip_uid'] + '_' + str(clip['clip_start_sec']) + '_' + str(clip['clip_end_sec']) for clip in clips]
            return clip_name

        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            ReduceExpandInputClips(
                                Compose(
                                    [
                                        Lambda(lambda x: x/255.0),
                                        Normalize(cfg.DATA.MEAN, cfg.DATA.STD)
                                    ]
                                    + video_transformer.random_scale_crop_flip(
                                        mode, cfg
                                    )
                                    + [video_transformer.uniform_temporal_subsample_repeated(cfg)]
                                )
                            ),
                        ]
                    ),
                ),
                Lambda(
                    lambda x: (
                        x["video"],
                        extract_forecast_labels(x),
                        extract_observed_labels(x),
                        extract_clip_id(x),
                        extract_forecast_times(x),
                        extract_clip_info(x)
                    )
                ),
            ]
        )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos



@DATASET_REGISTRY.register()
class Ego4dRecognitionLongerTimeSpan(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Ego4d ".format(mode)

        sampler = RandomSampler
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler

        clip_sampler_type = "uniform" if mode == "test" else "random"
        clip_duration = (
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        ) / self.cfg.DATA.TARGET_FPS
        if mode == "test":
            clip_sampler = CenterClipVideoSampler(clip_duration)
        else:
            clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)

        mode_ = 'test_unannotated' if mode =='test' else mode

        data_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'fho_lta_{mode_}.json')
        self.ann_file = data_path
        self.dataset = ptv_dataset_helper.clip_ar_auxdataset(
            data_path=data_path,
            clip_sampler=clip_sampler,
            num_input_clips=self.cfg.FORECASTING.NUM_INPUT_CLIPS,
            input_offset=self.cfg.FORECASTING.INPUT_OFFSET,
            video_sampler=sampler,
            decode_audio=False,
            transform=self._make_transform(mode, cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
        )
        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 2)
        )

    @property
    def sampler(self):
        return self.dataset.video_sampler

    def _make_transform(self, mode: str, cfg):
        class ReduceExpandInputClips:
            def __init__(self, transform):
                self.transform = transform

            def __call__(self, x):
                if x.dim() == 4:
                    x = x.unsqueeze(0)  # Handle num_clips=1

                n, c, t, h, w = x.shape
                x = x.transpose(0, 1)
                x = x.reshape(c, n * t, h, w)
                x = self.transform(x)

                if isinstance(x, list):
                    for i in range(len(x)):
                        c, _, h, w = x[i].shape
                        x[i] = x[i].reshape(c, n, -1, h, w)
                        x[i] = x[i].transpose(1, 0)
                else:
                    c, _, h, w = x.shape
                    x = x.reshape(c, n, t, h, w)
                    x = x.transpose(1, 0)

                return x

        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            ReduceExpandInputClips(
                                Compose(
                                    [
                                        Lambda(lambda x: x/255.0),
                                        Normalize(cfg.DATA.MEAN, cfg.DATA.STD)
                                    ]
                                    + video_transformer.random_scale_crop_flip(
                                        mode, cfg
                                    )
                                    + [video_transformer.uniform_temporal_subsample_repeated(cfg)]
                                )
                            ),
                        ]
                    ),
                ),
                Lambda(
                    lambda x: (
                        x["video"],
                        torch.tensor([x["ar_clips"][0]["verb_label"], x["ar_clips"][0]["noun_label"]]),
                        "a",
                        "b"
                    )
                ),
            ]
        )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos


@DATASET_REGISTRY.register()
class Ego4dLongTermAnticipationSequenceLabel(Ego4dLongTermAnticipation):
    def __init__(self, cfg, vocab, mode):
        super(Ego4dLongTermAnticipationSequenceLabel, self).__init__(cfg, mode)
        self.vocab = vocab
        self.verb_dict, self.noun_dict = map_label_to_action()
        self.lta_task_idx = vocab['action']  # to-do: add lta idx!
        self.eos_idx = vocab['</s>']

    def __getitem__(self, index):
        input, labels, a, b, c, d = next(self._dataset_iter)
        target_seq = [self.lta_task_idx]
        for label in labels:
            verb = self.verb_dict[label[0].item()]
            noun = self.noun_dict[label[1].item()]
            target_seq.append(self.vocab[verb])
            target_seq.append(self.vocab[noun])
        target_seq.append(self.eos_idx)
        # print(a, labels, verb, noun, target_seq)
        return input, torch.LongTensor(target_seq), a, b, c, d, labels


@DATASET_REGISTRY.register()
class Ego4dLongTermAnticipationSeparateSequenceLabel(Ego4dLongTermAnticipation):
    def __init__(self, cfg, vocab, mode):
        super(Ego4dLongTermAnticipationSeparateSequenceLabel, self).__init__(cfg, mode)
        self.vocab = vocab
        self.verb_dict, self.noun_dict = map_label_to_action()
        self.lta_verb_idx = vocab['lta_verb']
        self.lta_noun_idx = vocab['lta_noun']
        self.eos_idx = vocab['</s>']

    def __getitem__(self, index):
        inputs, labels, a, b, c, d = next(self._dataset_iter)
        target_seq_verb = [self.lta_verb_idx]
        target_seq_noun = [self.lta_noun_idx]
        for label in labels:
            verb = self.verb_dict[label[0].item()]
            noun = self.noun_dict[label[1].item()]
            target_seq_verb.append(self.vocab[verb])
            target_seq_noun.append(self.vocab[noun])
        target_seq_verb.append(self.eos_idx)
        target_seq_noun.append(self.eos_idx)
        # print(a, labels, verb, noun, target_seq)
        return inputs, torch.LongTensor(target_seq_verb), torch.LongTensor(target_seq_noun), \
               a, b, c, d, labels



class Ego4dARVisualize(torch.utils.data.Dataset):
    def __init__(self, cfg, video_info):
        self.cfg = cfg
        sampler = RandomSampler
        clip_sampler_type = "uniform"
        clip_duration = (self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE) / self.cfg.DATA.TARGET_FPS
        clip_sampler = CenterClipVideoSampler(clip_duration)
        print(f'clip sampler type {clip_sampler_type} clip duration {clip_duration}')
        self.clip_list = self._construct_clip_list(video_info)
        self.dataset = LabeledVideoDataset(
            self.clip_list,
            UntrimmedClipSampler(clip_sampler),
            sampler,
            make_transform_unlabeled("val", cfg),
            decode_audio=False,
            decoder="pyav",
        )
        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 1)
        )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos

    def _construct_clip_list(self, video_info):
        video_file = os.path.join("/checkpoint/sherryxue/ego4d/long_term_anticipation/clips", video_info["clip_id"]+".mp4")
        start_sec, end_sec = video_info['start_sec'], video_info['end_sec']
        temp_res = video_info['temp_res']
        t0 = start_sec - 2.93
        clip_list = []
        while t0 < end_sec - 8 + 2.93:
            clip_list.append((video_file, {
                "clip_uid": video_info["clip_id"],
                "clip_start_sec": t0,
                "clip_end_sec": t0 + 8,
            }))
            t0 = t0 + temp_res
        print(f'Start-End sec {start_sec}-{end_sec} | '
              f'Temporal resolution {temp_res} seconds | List len {len(clip_list)}')
        return clip_list