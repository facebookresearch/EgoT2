#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import json
import time
import cv2
import av
import numpy as np
import torch
from iopath.common.file_io import g_pathmgr

from .build import DATASET_REGISTRY
from utils.pnr.trim import _get_frames
from .long_term_anticipation import Ego4dRecognition

@DATASET_REGISTRY.register()
class Ego4dRecognitionwithAuxTask(Ego4dRecognition):
    def __init__(self, cfg, cfg_aux, mode):
        super(Ego4dRecognitionwithAuxTask, self).__init__(cfg, mode)
        self.mode = mode
        self.cfg_aux = cfg_aux
        self.input_dir = '/datasets01/ego4d_track2/v1/clips/'
        self.output_dir = '/checkpoint/sherryxue/ego4d/features_fho_forAction/'
        # self.package = dict()
        # self._construct_pnr_list()
        self._construct_pnr_mapping()

    def _construct_pnr_list(self):
        with g_pathmgr.open(self.ann_file, "r") as f:
            annotations = json.load(f)
        cnt = 0
        name_dict = dict()
        for entry in annotations['clips']:
            unique_name = entry['clip_uid'] + '_' + str(entry['action_clip_start_frame']) + '_' +\
                       str(entry['action_clip_end_frame']) + '_' + str(entry['action_idx'])
            assert unique_name not in name_dict
            name_dict[unique_name] = entry
            self.package[cnt] = {
                'unique_id': unique_name,
                'clip_uid': entry['clip_uid'],
                'clip_start_frame': entry['action_clip_start_frame'],
                'clip_end_frame': entry['action_clip_end_frame'],
                'clip_start_sec': entry['action_clip_start_sec'],
                'clip_end_sec': entry['action_clip_end_sec'],
                'pnr_frame': None
            }
            cnt = cnt + 1
        print(f'Number of clips for {self.mode} (PNR/OSCC): {len(self.package)}')

    def _construct_pnr_mapping(self):
        with g_pathmgr.open(self.ann_file, "r") as f:
            annotations = json.load(f)
        self.pnr_mapping = dict()
        for entry in annotations['clips']:
            key_name = entry['clip_uid'] + '_' + str(entry['action_clip_start_sec']) + '_' + str(entry['action_clip_end_sec'])
            unique_name = entry['clip_uid'] + '_' + str(entry['action_clip_start_frame']) + '_' +\
                       str(entry['action_clip_end_frame']) + '_' + str(entry['action_idx'])
            # if key_name in self.pnr_mapping:
            #     print(f'{key_name} already exists')
            self.pnr_mapping[key_name] = {
                'unique_id': unique_name,
                'clip_uid': entry['clip_uid'],
                'clip_start_frame': entry['action_clip_start_frame'],
                'clip_end_frame': entry['action_clip_end_frame'],
                'clip_start_sec': entry['action_clip_start_sec'],
                'clip_end_sec': entry['action_clip_end_sec'],
                'pnr_frame': None
            }
        print(f'Number of clips for {self.mode} (PNR/OSCC): {len(self.pnr_mapping)}')

    def __getitem__(self, index):
        # return self._get_item_pnr(index)
        value = next(self._dataset_iter)
        video_info = value[3]
        name = video_info['video_name'].replace('.mp4', '') + '_' + str(video_info['start_sec']) + '_' + str(video_info['end_sec'])
        assert name in self.pnr_mapping
        return {'orig': value, 'pnr': self._get_item_pnr(name)}

    def _get_item_pnr(self, name):
        info = self.pnr_mapping[name]
        # info = self.package[index]
        self._extract_clip_frames(info)
        frames, labels, _ = self._sample_frames_gen_labels(info)
        frames = torch.as_tensor(frames).permute(3, 0, 1, 2)
        return [frames]
        # return [frames], info['clip_uid']

    def _extract_clip_frames(self, info):
        """
        This method is used to extract and save frames for all the 8 seconds
        clips. If the frames are already saved, it does nothing.
        """
        clip_start_frame = info['clip_start_frame']
        clip_end_frame = info['clip_end_frame']
        video_path = os.path.join(self.input_dir, info['clip_uid']+'.mp4')

        assert os.path.exists(video_path)
        clip_save_path = os.path.join(self.output_dir, info['unique_id'])

        if os.path.isdir(clip_save_path):
            # The frames for this clip are already saved.
            num_frames = len(os.listdir(clip_save_path))
            if num_frames < (clip_end_frame - clip_start_frame):
                print(
                    f'Deleting {clip_save_path} as it has {num_frames} frames'
                )
                os.system(f'rm -r {clip_save_path}')
            else:
                return None
        print(f'Saving frames for {clip_save_path}...')
        os.makedirs(clip_save_path)

        start = time.time()
        # We need to save the frames for this clip.
        frames_list = [
            i for i in range(clip_start_frame, clip_end_frame + 1, 1)
        ]
        frames = self.get_frames_for(
            video_path,
            frames_list,
        )
        desired_shorter_side = 384
        num_saved_frames = 0
        for frame, frame_count in zip(frames, frames_list):
            original_height, original_width, _ = frame.shape
            if original_height < original_width:
                # Height is the shorter side
                new_height = desired_shorter_side
                new_width = np.round(
                    original_width*(desired_shorter_side/original_height)
                ).astype(np.int32)
            elif original_height > original_width:
                # Width is the shorter side
                new_width = desired_shorter_side
                new_height = np.round(
                    original_height*(desired_shorter_side/original_width)
                ).astype(np.int32)
            else:
                # Both are the same
                new_height = desired_shorter_side
                new_width = desired_shorter_side
            assert np.isclose(
                new_width/new_height,
                original_width/original_height,
                0.01
            )
            frame = cv2.resize(
                frame,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(
                os.path.join(
                    clip_save_path,
                    f'{frame_count}.jpeg'
                ),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            num_saved_frames += 1
        # print(f'Time taken: {time.time() - start}; {num_saved_frames} '
        #     f'frames saved; {clip_save_path}')
        return None

    def _sample_frames(
        self,
        clip_start_frame,
        clip_end_frame,
        num_frames_required,
        pnr_frame
    ):
        num_frames = clip_end_frame - clip_start_frame
        if num_frames < num_frames_required:
            print(f'Issue: {num_frames}; {num_frames_required}')
        error_message = "Can\'t sample more frames than there are in the video"
        assert num_frames >= num_frames_required, error_message
        lower_lim = np.floor(num_frames/num_frames_required)
        upper_lim = np.ceil(num_frames/num_frames_required)
        lower_frames = list()
        upper_frames = list()
        lower_keyframe_candidates_list = list()
        upper_keyframe_candidates_list = list()
        for frame_count in range(clip_start_frame, clip_end_frame, 1):
            if frame_count % lower_lim == 0:
                lower_frames.append(frame_count)
                if pnr_frame is not None:
                    lower_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    lower_keyframe_candidates_list.append(0.0)
            if frame_count % upper_lim == 0:
                upper_frames.append(frame_count)
                if pnr_frame is not None:
                    upper_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    upper_keyframe_candidates_list.append(0.0)
        if len(upper_frames) < num_frames_required:
            return (
                lower_frames[:num_frames_required],
                lower_keyframe_candidates_list[:num_frames_required]
            )
        return (
            upper_frames[:num_frames_required],
            upper_keyframe_candidates_list[:num_frames_required]
        )

    def _load_frame(self, frame_path):
        """
        This method is used to read a frame and do some pre-processing.

        Args:
            frame_path (str): Path to the frame

        Returns:
            frames (ndarray): Image as a numpy array
        """
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (
            self.cfg_aux.DATA.CROP_SIZE,
            self.cfg_aux.DATA.CROP_SIZE
        ))
        frame = np.expand_dims(frame, axis=0).astype(np.float32)
        return frame

    def _sample_frames_gen_labels(self, info):
        clip_path = os.path.join(self.output_dir, info['unique_id'])
        message = f'Clip path {clip_path} does not exists...'
        assert os.path.isdir(clip_path), message

        num_frames_per_video = (
            self.cfg_aux.DATA.SAMPLING_FPS * self.cfg_aux.DATA.CLIP_LEN_SEC
        )
        pnr_frame = info['pnr_frame']
        if self.mode == 'train':
            # Random clipping
            # Randomly choosing the duration of clip (between 5-8 seconds)
            random_length_seconds = np.random.uniform(5, 8)
            random_start_seconds = info['clip_start_sec'] + np.random.uniform(
                8 - random_length_seconds
            )
            random_start_frame = np.floor(
                random_start_seconds * 30
            ).astype(np.int32)
            random_end_seconds = random_start_seconds + random_length_seconds
            if random_end_seconds > info['clip_end_sec']:
                random_end_seconds = info['clip_end_sec']
            random_end_frame = np.floor(
                random_end_seconds * 30
            ).astype(np.int32)
            if pnr_frame is not None:
                keyframe_after_end = pnr_frame > random_end_frame
                keyframe_before_start = pnr_frame < random_start_frame
                if keyframe_after_end:
                    random_end_frame = info['clip_end_frame']
                if keyframe_before_start:
                    random_start_frame = info['clip_start_frame']
        elif self.mode in ['test', 'val']:
            random_start_frame = info['clip_start_frame']
            random_end_frame = info['clip_end_frame']

        if pnr_frame is not None:
            message = (f'Random start frame {random_start_frame} Random end '
                f'frame {random_end_frame} info {info} clip path {clip_path}')
            assert random_start_frame <= pnr_frame <= random_end_frame, message
        else:
            message = (f'Random start frame {random_start_frame} Random end '
                f'frame {random_end_frame} info {info} clip path {clip_path}')
            assert random_start_frame < random_end_frame, message

        candidate_frame_nums, keyframe_candidates_list = self._sample_frames(
            random_start_frame,
            random_end_frame,
            num_frames_per_video,
            pnr_frame
        )
        frames = list()
        for frame_num in candidate_frame_nums:
            frame_path = os.path.join(clip_path, f'{frame_num}.jpeg')
            message = f'{frame_path}; {candidate_frame_nums}'
            assert os.path.isfile(frame_path), message
            frames.append(self._load_frame(frame_path))
        if pnr_frame is not None:
            keyframe_location = np.argmin(keyframe_candidates_list)
            hard_labels = np.zeros(len(candidate_frame_nums))
            hard_labels[keyframe_location] = 1
            labels = hard_labels
        else:
            labels = keyframe_candidates_list
        # Calculating the effective fps. In other words, the fps after sampling
        # changes when we are randomly clipping and varying the duration of the
        # clip
        final_clip_length = (random_end_frame/30) - (random_start_frame/30)
        effective_fps = num_frames_per_video / final_clip_length
        return np.concatenate(frames), np.array(labels), effective_fps

    def get_frames_for(self, video_path, frames_list):
        """
        Code for decoding the video
        """
        frames = list()
        with av.open(video_path) as container:
            for frame in _get_frames(
                frames_list,
                container,
                include_audio=False,
                audio_buffer_frames=0
            ):
                frame = frame.to_rgb().to_ndarray()
                frames.append(frame)
        return frames


from utils.multitask.build_vocab import map_label_to_action
@DATASET_REGISTRY.register()
class Ego4dRecognitionwithAuxTaskSequenceLabel(Ego4dRecognitionwithAuxTask):
    def __init__(self, cfg, cfg_aux, vocab, mode):
        super(Ego4dRecognitionwithAuxTaskSequenceLabel, self).__init__(cfg, cfg_aux, mode)
        self.vocab = vocab
        self.verb_dict, self.noun_dict = map_label_to_action()
        self.action_task_idx = vocab['action']
        self.eos_idx = vocab['</s>']

    def __getitem__(self, index):
        inputs, labels, a, video_info = next(self._dataset_iter)
        verb = self.verb_dict[labels[0].item()]
        noun = self.noun_dict[labels[1].item()]
        target_seq = [self.action_task_idx, self.vocab[verb], self.vocab[noun], self.eos_idx]
        # print(a, labels, verb, noun, target_seq)
        orig_new = inputs, torch.LongTensor(target_seq), a, video_info, labels

        name = video_info['video_name'].replace('.mp4', '') + '_' + str(video_info['start_sec']) + '_' + str(
            video_info['end_sec'])
        assert name in self.pnr_mapping
        return {'orig': orig_new, 'pnr': self._get_item_pnr(name)}


@DATASET_REGISTRY.register()
class Ego4dRecognitionwithAuxTaskSeparateSequenceLabel(Ego4dRecognitionwithAuxTask):
    def __init__(self, cfg, cfg_aux, vocab, mode):
        super(Ego4dRecognitionwithAuxTaskSeparateSequenceLabel, self).__init__(cfg, cfg_aux, mode)
        self.vocab = vocab
        self.verb_dict, self.noun_dict = map_label_to_action()
        self.action_verb_idx = vocab['action_verb']
        self.action_noun_idx = vocab['action_noun']
        self.eos_idx = vocab['</s>']

    def __getitem__(self, index):
        inputs, labels, a, video_info = next(self._dataset_iter)
        verb = self.verb_dict[labels[0].item()]
        noun = self.noun_dict[labels[1].item()]
        target_seq_verb = [self.action_verb_idx, self.vocab[verb], self.eos_idx]
        target_seq_noun = [self.action_noun_idx, self.vocab[noun], self.eos_idx]
        orig_new = inputs, torch.LongTensor(target_seq_verb), torch.LongTensor(target_seq_noun), a, video_info, labels

        name = video_info['video_name'].replace('.mp4', '') + '_' + str(video_info['start_sec']) + '_' + str(
            video_info['end_sec'])
        assert name in self.pnr_mapping
        return {'orig': orig_new, 'pnr': self._get_item_pnr(name)}




from .long_term_anticipation import Ego4dARVisualize
class Ego4dARVisualizewithAuxTask(Ego4dARVisualize):
    def __init__(self, cfg, cfg_aux, video_info):
        super(Ego4dARVisualizewithAuxTask, self).__init__(cfg, video_info)
        self.cfg_aux = cfg_aux
        self.input_dir = '/datasets01/ego4d_track2/v1/clips/'
        self.output_dir = '/checkpoint/sherryxue/ego4d/features_fho_visualize/'
        os.makedirs(self.output_dir, exist_ok=True)
        self.mode = "val"
        self._construct_pnr_mapping()

    def _construct_pnr_mapping(self):
        self.pnr_mapping = dict()
        for clip in self.clip_list:
            video_file, entry = clip
            key_name = entry['clip_uid'] + '_' + str(entry['clip_start_sec']) + '_' + str(entry['clip_end_sec'])
            self.pnr_mapping[key_name] = {
                'unique_id': key_name,
                'clip_uid': entry['clip_uid'],
                'clip_start_frame': int(entry['clip_start_sec'] * 30),
                'clip_end_frame': int(entry['clip_end_sec'] * 30),
                'clip_start_sec': entry['clip_start_sec'],
                'clip_end_sec': entry['clip_end_sec'],
                'pnr_frame': None
            }
        print(f'Number of clips (PNR/OSCC): {len(self.pnr_mapping)}')

    def __getitem__(self, index):
        # return self._get_item_pnr(index)
        value = next(self._dataset_iter)
        video_info = value[1]
        name = video_info['video_name'].replace('.mp4', '') + '_' + str(video_info['start_sec']) + '_' + str(video_info['end_sec'])
        assert name in self.pnr_mapping
        return {'orig': value, 'pnr': self._get_item_pnr(name)}

    def _get_item_pnr(self, name):
        info = self.pnr_mapping[name]
        # info = self.package[index]
        self._extract_clip_frames(info)
        frames, labels, _ = self._sample_frames_gen_labels(info)
        frames = torch.as_tensor(frames).permute(3, 0, 1, 2)
        return [frames]
        # return [frames], info['clip_uid']

    def _extract_clip_frames(self, info):
        """
        This method is used to extract and save frames for all the 8 seconds
        clips. If the frames are already saved, it does nothing.
        """
        clip_start_frame = info['clip_start_frame']
        clip_end_frame = info['clip_end_frame']
        video_path = os.path.join(self.input_dir, info['clip_uid']+'.mp4')

        assert os.path.exists(video_path)
        clip_save_path = os.path.join(self.output_dir, info['unique_id'])

        if os.path.isdir(clip_save_path):
            # The frames for this clip are already saved.
            num_frames = len(os.listdir(clip_save_path))
            if num_frames < (clip_end_frame - clip_start_frame):
                print(
                    f'Deleting {clip_save_path} as it has {num_frames} frames'
                )
                os.system(f'rm -r {clip_save_path}')
            else:
                return None
        print(f'Saving frames for {clip_save_path}...')
        os.makedirs(clip_save_path)

        start = time.time()
        # We need to save the frames for this clip.
        frames_list = [
            i for i in range(clip_start_frame, clip_end_frame + 1, 1)
        ]
        frames = self.get_frames_for(
            video_path,
            frames_list,
        )
        desired_shorter_side = 384
        num_saved_frames = 0
        for frame, frame_count in zip(frames, frames_list):
            original_height, original_width, _ = frame.shape
            if original_height < original_width:
                # Height is the shorter side
                new_height = desired_shorter_side
                new_width = np.round(
                    original_width*(desired_shorter_side/original_height)
                ).astype(np.int32)
            elif original_height > original_width:
                # Width is the shorter side
                new_width = desired_shorter_side
                new_height = np.round(
                    original_height*(desired_shorter_side/original_width)
                ).astype(np.int32)
            else:
                # Both are the same
                new_height = desired_shorter_side
                new_width = desired_shorter_side
            assert np.isclose(
                new_width/new_height,
                original_width/original_height,
                0.01
            )
            frame = cv2.resize(
                frame,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(
                os.path.join(
                    clip_save_path,
                    f'{frame_count}.jpeg'
                ),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            num_saved_frames += 1
        # print(f'Time taken: {time.time() - start}; {num_saved_frames} '
        #     f'frames saved; {clip_save_path}')
        return None

    def _sample_frames(
        self,
        clip_start_frame,
        clip_end_frame,
        num_frames_required,
        pnr_frame
    ):
        num_frames = clip_end_frame - clip_start_frame
        if num_frames < num_frames_required:
            print(f'Issue: {num_frames}; {num_frames_required}')
        error_message = "Can\'t sample more frames than there are in the video"
        assert num_frames >= num_frames_required, error_message
        lower_lim = np.floor(num_frames/num_frames_required)
        upper_lim = np.ceil(num_frames/num_frames_required)
        lower_frames = list()
        upper_frames = list()
        lower_keyframe_candidates_list = list()
        upper_keyframe_candidates_list = list()
        for frame_count in range(clip_start_frame, clip_end_frame, 1):
            if frame_count % lower_lim == 0:
                lower_frames.append(frame_count)
                if pnr_frame is not None:
                    lower_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    lower_keyframe_candidates_list.append(0.0)
            if frame_count % upper_lim == 0:
                upper_frames.append(frame_count)
                if pnr_frame is not None:
                    upper_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    upper_keyframe_candidates_list.append(0.0)
        if len(upper_frames) < num_frames_required:
            return (
                lower_frames[:num_frames_required],
                lower_keyframe_candidates_list[:num_frames_required]
            )
        return (
            upper_frames[:num_frames_required],
            upper_keyframe_candidates_list[:num_frames_required]
        )

    def _load_frame(self, frame_path):
        """
        This method is used to read a frame and do some pre-processing.

        Args:
            frame_path (str): Path to the frame

        Returns:
            frames (ndarray): Image as a numpy array
        """
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (
            self.cfg_aux.DATA.CROP_SIZE,
            self.cfg_aux.DATA.CROP_SIZE
        ))
        frame = np.expand_dims(frame, axis=0).astype(np.float32)
        return frame

    def _sample_frames_gen_labels(self, info):
        clip_path = os.path.join(self.output_dir, info['unique_id'])
        message = f'Clip path {clip_path} does not exists...'
        assert os.path.isdir(clip_path), message

        num_frames_per_video = (
            self.cfg_aux.DATA.SAMPLING_FPS * self.cfg_aux.DATA.CLIP_LEN_SEC
        )
        pnr_frame = info['pnr_frame']
        if self.mode == 'train':
            # Random clipping
            # Randomly choosing the duration of clip (between 5-8 seconds)
            random_length_seconds = np.random.uniform(5, 8)
            random_start_seconds = info['clip_start_sec'] + np.random.uniform(
                8 - random_length_seconds
            )
            random_start_frame = np.floor(
                random_start_seconds * 30
            ).astype(np.int32)
            random_end_seconds = random_start_seconds + random_length_seconds
            if random_end_seconds > info['clip_end_sec']:
                random_end_seconds = info['clip_end_sec']
            random_end_frame = np.floor(
                random_end_seconds * 30
            ).astype(np.int32)
            if pnr_frame is not None:
                keyframe_after_end = pnr_frame > random_end_frame
                keyframe_before_start = pnr_frame < random_start_frame
                if keyframe_after_end:
                    random_end_frame = info['clip_end_frame']
                if keyframe_before_start:
                    random_start_frame = info['clip_start_frame']
        elif self.mode in ['test', 'val']:
            random_start_frame = info['clip_start_frame']
            random_end_frame = info['clip_end_frame']

        if pnr_frame is not None:
            message = (f'Random start frame {random_start_frame} Random end '
                f'frame {random_end_frame} info {info} clip path {clip_path}')
            assert random_start_frame <= pnr_frame <= random_end_frame, message
        else:
            message = (f'Random start frame {random_start_frame} Random end '
                f'frame {random_end_frame} info {info} clip path {clip_path}')
            assert random_start_frame < random_end_frame, message

        candidate_frame_nums, keyframe_candidates_list = self._sample_frames(
            random_start_frame,
            random_end_frame,
            num_frames_per_video,
            pnr_frame
        )
        frames = list()
        for frame_num in candidate_frame_nums:
            frame_path = os.path.join(clip_path, f'{frame_num}.jpeg')
            message = f'{frame_path}; {candidate_frame_nums}'
            assert os.path.isfile(frame_path), message
            frames.append(self._load_frame(frame_path))
        if pnr_frame is not None:
            keyframe_location = np.argmin(keyframe_candidates_list)
            hard_labels = np.zeros(len(candidate_frame_nums))
            hard_labels[keyframe_location] = 1
            labels = hard_labels
        else:
            labels = keyframe_candidates_list
        # Calculating the effective fps. In other words, the fps after sampling
        # changes when we are randomly clipping and varying the duration of the
        # clip
        final_clip_length = (random_end_frame/30) - (random_start_frame/30)
        effective_fps = num_frames_per_video / final_clip_length
        return np.concatenate(frames), np.array(labels), effective_fps

    def get_frames_for(self, video_path, frames_list):
        """
        Code for decoding the video
        """
        frames = list()
        with av.open(video_path) as container:
            for frame in _get_frames(
                frames_list,
                container,
                include_audio=False,
                audio_buffer_frames=0
            ):
                frame = frame.to_rgb().to_ndarray()
                frames.append(frame)
        return frames