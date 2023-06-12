#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, cv2, json, glob, logging, soundfile, python_speech_features
import torch
import torchvision.transforms as transforms
import numpy as np
import random
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import interp1d
from collections import defaultdict, OrderedDict


logger = logging.getLogger(__name__)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def helper():
    return defaultdict(OrderedDict)


def check(track):
    inter_track = []
    framenum = []
    bboxes = []
    for frame in track:
        x = frame['x']
        y = frame['y']
        w = frame['width']
        h = frame['height']
        if (w <= 0 or h <= 0 or 
            frame['frameNumber']==0 or
            len(frame['Person ID'])==0):
            continue
        framenum.append(frame['frameNumber'])
        x = max(x, 0)
        y = max(y, 0)
        bbox = [x, y, x + w, y + h]
        bboxes.append(bbox)
    
    if len(framenum) == 0:
        return inter_track

    framenum = np.array(framenum)
    bboxes = np.array(bboxes)

    gt_frames = framenum[-1] - framenum[0] + 1

    frame_i = np.arange(framenum[0], framenum[-1]+1)

    if gt_frames > framenum.shape[0]:
        bboxes_i = []
        for ij in range(0,4):
            interpfn  = interp1d(framenum, bboxes[:,ij])
            bboxes_i.append(interpfn(frame_i))
        bboxes_i  = np.stack(bboxes_i, axis=1)
    else:
        frame_i = framenum
        bboxes_i = bboxes

    #assemble new tracklet
    template = track[0]
    for i, (frame, bbox) in enumerate(zip(frame_i, bboxes_i)):
        record = template.copy()
        record['frameNumber'] = frame
        record['x'] = bbox[0]
        record['y'] = bbox[1]
        record['width'] = bbox[2] - bbox[0]
        record['height'] = bbox[3] - bbox[1]
        inter_track.append(record)
    return inter_track


def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples


def get_bbox(uid, json_path):
    bboxes = {}
    vid_json_dir = os.path.join(json_path, uid)
    tracklets = glob.glob(f'{vid_json_dir}/*.json')
    for idx, t in enumerate(tracklets):
        with open(t, 'r') as j:
            frames = json.load(j)

        # check the bbox, interpolate when necessary
        frames = check(frames)

        for frame in frames:
            frameid = frame['frameNumber']
            bbox = (frame['x'],
                    frame['y'],
                    frame['x'] + frame['width'],
                    frame['y'] + frame['height'])
            identifier = str(frameid) + ':' + frame['Person ID']
            bboxes[identifier] = bbox

    return bboxes


def make_dataset(file_list, img_anno, audio_anno, stride=1, min_frames=15, max_frames=150):

    logger.info('load: ' + file_list)
    face_crop = {}
    segments = []

    with open(file_list, 'r') as f:
        videos = f.readlines()

    for uid in videos:
        uid = uid.strip()
        face_crop[uid] = get_bbox(uid, img_anno)

        with open(os.path.join(audio_anno, uid + '.json'), ) as js:
            gts = json.load(js)

        for idx, gt in enumerate(gts):
            if 'tags' not in gt:
                personid = gt['label']
                label = 0
                start_frame = int(gt['start_frame'])
                end_frame = int(gt['end_frame'])
                seg_length = end_frame - start_frame + 1

            else:
                personid = gt['label']
                label = 1
                start_frame = int(gt['start_frame'])
                end_frame = int(gt['end_frame'])
                seg_length = end_frame - start_frame + 1

            if ('train' in file_list and seg_length < min_frames) or (seg_length <= 1) or (personid == 0):
                continue
            elif seg_length > max_frames:
                it = int(seg_length / max_frames)
                for i in range(it):
                    sub_start = start_frame + i*max_frames
                    sub_end = min(end_frame, sub_start + max_frames)
                    sub_length = sub_end - sub_start + 1
                    if sub_length < min_frames:
                        continue
                    segments.append([uid, personid, label, sub_start, sub_end, idx])
            else:
                segments.append([uid, personid, label, start_frame, end_frame, idx])
    return segments, face_crop


class ImagerLoader2Task(torch.utils.data.Dataset):
    def __init__(self, img_path, audio_path, file_list, img_json, audio_json,
                 stride=1, mode='train', transform=None):
        self.img_path = img_path
        assert os.path.exists(self.img_path), 'image path not exist'
        self.audio_path = audio_path
        assert os.path.exists(self.audio_path), 'audio path not exist'
        self.file_list = file_list
        assert os.path.exists(self.file_list), f'{mode} list not exist'
        self.img_json = img_json
        assert os.path.exists(self.img_json), 'json path not exist'
        self.audio_json = audio_json
        assert os.path.exists(self.audio_json), 'talking to me path not exist'
        segments, face_crop = make_dataset(file_list, img_json, audio_json, stride=stride)

        avg_frame = []
        for segment in segments:
            numframe = segment[4] - segment[3]
            avg_frame.append(int(numframe))
        print('Data num', len(segments), 'Average num frames', np.mean(avg_frame))

        self.segments = segments
        self.face_crop = face_crop
        self.transform = transform
        self.mode = mode

    def __getitem__(self, indices):
        video, video_asd = self._get_video(indices, debug=False)
        audio, audio_asd = self._get_audio(indices, video_asd.shape[0])
        # print(video.shape, video_asd.shape, audio.shape, audio_asd.shape)
        return video, video_asd, audio, audio_asd, self._get_target(indices)

    def __len__(self):
        return len(self.segments)

    def _get_video(self, index, debug=False):
        uid, personid, _, start_frame, end_frame, _ = self.segments[index]
        video = []
        frames = []
        frame_name = []
        for i in range(start_frame, end_frame + 1):
            key = str(i) + ':' + str(personid)
            if key in self.face_crop[uid]:
                bbox = self.face_crop[uid][key]
                img = f'{self.img_path}/{uid}/img_{i:05d}.jpg'
                frames.append(bbox)
                frame_name.append(img)
                if not os.path.exists(img):
                    video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                    continue
                assert os.path.exists(img), f'img: {img} not found'
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                face = img[y1: y2, x1: x2, :]
                try:
                    face = cv2.resize(face, (224, 224))
                except:
                    # bad bbox
                    face = np.zeros((224, 224, 3), dtype=np.uint8)

                if debug:
                    import matplotlib.pyplot as plt
                    plt.imshow(face)
                    plt.axis('off')
                    plt.show()
                    
                video.append(np.expand_dims(face, axis=0))
            else:
                video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                continue
        video = np.concatenate(video, axis=0)  # (numframes, 224, 224, 3)
        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)

        dets = {'x': [], 'y': [], 's': []}
        for bbox in frames:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            dets['s'].append(max((y2 - y1), (x2 - x1)) / 2)
            dets['y'].append((y2 + y1) / 2)  # crop center x
            dets['x'].append((x2 + x1) / 2)  # crop center y
        kernel_size = min((len(dets['s']) - len(dets['s']) % 2 + 1), 13)
        dets['s'] = signal.medfilt(dets['s'], kernel_size=kernel_size)  # Smooth detections
        dets['x'] = np.array(dets['x'])
        dets['x'][1:] = dets['x'][:-1] * 0.8 + dets['x'][1:] * 0.2
        dets['y'] = np.array(dets['y'])
        dets['y'][1:] = dets['y'][:-1] * 0.8 + dets['y'][1:] * 0.2

        faces = []
        H = 112
        # no training-time augmentation for now
        for i, img_name in enumerate(frame_name):
            if not os.path.exists(img_name):
                faces.append(np.zeros((H, H), dtype=np.uint8))
                continue
            img = cv2.imread(img_name)
            cs = 0.4
            bs = dets['s'][i]
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
            img = np.pad(img, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my = dets['y'][i] + bsi  # BBox center Y
            mx = dets['x'][i] + bsi  # BBox center X
            face = img[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face1 = cv2.resize(face, (2 * H, 2 * H))  # (224, 224)
            face = face1[int(112 - (112 / 2)):int(112 + (112 / 2)),
                   int(112 - (112 / 2)):int(112 + (112 / 2))]  # (112, 112)
            # print('img shape', face.shape)
            if debug:
                import matplotlib.pyplot as plt
                plt.subplot(1, 2, 1)
                plt.imshow(np.asarray(face), cmap='gray', vmin=0, vmax=255)
                plt.subplot(1, 2, 2)
                plt.imshow(np.asarray(face1), cmap='gray', vmin=0, vmax=255)
                plt.axis('off')
                plt.show()
            faces.append(face)

        if len(faces) == 0:  # mask
            video_asd = torch.zeros((video.shape[0], H, H))
        else:
            video_asd = torch.FloatTensor(np.array(faces))
        # print(video.shape, video_asd.shape)
        return video, video_asd

    def _get_audio(self, index, numFrames):
        uid, _, _, start_frame, end_frame, _ = self.segments[index]
        audio, sample_rate = soundfile.read(f'{self.audio_path}/{uid}.wav')
        onset = int(start_frame / 30 * sample_rate)
        offset = int(end_frame / 30 * sample_rate)
        crop_audio = normalize(audio[onset: offset])
        audio = torch.FloatTensor(crop_audio)

        fps = 30
        if crop_audio.shape[0] == 0:
            return audio, torch.zeros((numFrames * 4, 13))

        audio_asd = python_speech_features.mfcc(crop_audio, 16000, numcep=13, winlen=0.025 * 25 / fps, winstep=0.010 * 25 / fps)
        maxAudio = int(numFrames * 4)
        if audio_asd.shape[0] < maxAudio:
            shortage = maxAudio - audio_asd.shape[0]
            audio_asd = np.pad(audio_asd, ((0, shortage), (0, 0)), 'wrap')
        audio_asd = audio_asd[:maxAudio, :]
        # print('maxaudio', maxAudio, audio_asd.shape)
        audio_asd = torch.FloatTensor(np.array(audio_asd))
        return audio, audio_asd

    def _get_target(self, index):
        if self.mode == 'train':
            return torch.LongTensor([self.segments[index][2]])
        else:
            return self.segments[index]


class ImagerSeqLoader2Task(ImagerLoader2Task):
    def __init__(self, vocab, img_path, audio_path, file_list, img_json, audio_json,
                 stride=1, mode='train', transform=None):
        super(ImagerSeqLoader2Task, self).__init__(img_path, audio_path, file_list, img_json, audio_json,
                 stride, mode, transform)
        self.vocab = vocab
        self.task_idx = vocab['ttm']
        self.eos_idx = vocab['</s>']

    def _get_target(self, index):
        label = self.segments[index][2]
        target_seq = [self.task_idx, self.vocab[str(label)], self.eos_idx]
        if self.mode == 'train':
            return torch.LongTensor(target_seq)
        else:
            return self.segments[index], torch.LongTensor(target_seq)


def make_test_dataset(data_path, seg_info, min_frames=15, max_frames=150):
    logger.info('load: ' + data_path)
    segments = []

    for sid in tqdm(os.listdir(data_path)):
        aud_path = os.path.join(data_path, sid, 'audio', 'aud.wav')
        if os.path.exists(os.path.join(data_path, sid, 'face')):
            vid_path = os.path.join(data_path, sid, 'face')
        else:
            vid_path = 'None'
        seg_length = seg_info[sid]['frame_num']
        start_frame = 0
        end_frame = seg_length - 1
        if seg_length > max_frames:
            it = int(seg_length / max_frames)
            for i in range(it):
                sub_start = start_frame + i * max_frames
                sub_end = min(end_frame, sub_start + max_frames)
                sub_length = sub_end - sub_start + 1
                if sub_length < min_frames:
                    continue
                segments.append([sid, aud_path, vid_path, sub_start, sub_end])
        else:
            segments.append([sid, aud_path, vid_path, start_frame, end_frame])
    return segments


class TestImagerLoader2Task(torch.utils.data.Dataset):
    def __init__(self, data_path, seg_info, transform=None):
        self.data_path = data_path
        assert os.path.exists(self.data_path), 'image path not exist'

        self.seg_info = json.load(open(seg_info))

        self.segments = make_test_dataset(data_path, self.seg_info)
        self.transform = transform

    def __getitem__(self, indices):
        video, video_asd = self._get_video(indices, debug=False)
        audio, audio_asd = self._get_audio(indices, video_asd.shape[0])
        sid = self.segments[indices][0]
        return video, video_asd, audio, audio_asd, sid, self.seg_info[sid]['frame_list']

    def __len__(self):
        return len(self.segments)

    def _get_video(self, index, debug=False):
        sid, _, vid_path, start_frame, end_frame = self.segments[index]
        video = []
        faces = []
        H = 112
        if os.path.exists(vid_path):
            fid2path = {}
            for img_path in os.listdir(vid_path):
                fid = int(img_path.split('.')[0])
                fid2path[fid] = os.path.join(vid_path, img_path)

            for fid in range(start_frame, end_frame + 1):
                if fid in fid2path.keys():
                    img_orig = cv2.imread(fid2path[fid])
                    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                    face = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (2 * H, 2 * H))  # (224, 224)
                    face = face[int(112 - (112 / 2)):int(112 + (112 / 2)),
                           int(112 - (112 / 2)):int(112 + (112 / 2))]  # (112, 112)
                else:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                    face = np.zeros((112, 112), dtype=np.uint8)

                if debug:
                    import matplotlib.pyplot as plt
                    plt.imshow(img)
                    plt.show()

                video.append(np.expand_dims(img, axis=0))
                faces.append(face)

        else:
            for fid in range(self.frame_num[index]):
                video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                faces.append(np.zeros((1, 112, 112), dtype=np.uint8))

        video = np.concatenate(video, axis=0)
        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)
        video_asd = torch.FloatTensor(np.array(faces))
        return video, video_asd

    def _get_audio(self, index, numFrames):
        sid, aud_path, _, start_frame, end_frame = self.segments[index]
        audio, sample_rate = soundfile.read(aud_path)
        onset = int(start_frame / 30 * sample_rate)
        offset = int(end_frame / 30 * sample_rate)
        crop_audio = normalize(audio[onset: offset])
        fps = 30
        audio_asd = python_speech_features.mfcc(crop_audio, 16000, numcep=13, winlen=0.025 * 25 / fps,
                                                winstep=0.010 * 25 / fps)
        maxAudio = int(numFrames * 4)
        if audio_asd.shape[0] < maxAudio:
            shortage = maxAudio - audio_asd.shape[0]
            audio_asd = np.pad(audio_asd, ((0, shortage), (0, 0)), 'wrap')
        audio_asd = audio_asd[:maxAudio, :]
        audio_asd = torch.FloatTensor(np.array(audio_asd))
        return torch.tensor(crop_audio, dtype=torch.float32), audio_asd