#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
from pytorch_lightning.core import LightningModule
from dataset.ttm.data_loader_2task import ImagerLoader2Task, TestImagerLoader2Task
from dataset.ttm.sampler import SequenceBatchSampler
from models.ttm.build import build_model
from utils.ttm.utils import PostProcessor, test_PostProcessor, get_transform, collate_fn_2task, pred2json


class TalkingToMe2Loader(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.checkpoint_metric = "val_mAP"
        self.model = build_model(args)
        class_weights = torch.FloatTensor(args.weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.postprocess = PostProcessor(self.args)
        self.test_postprocess = test_PostProcessor(self.args)

    def on_train_epoch_start(self):
        self.train_loader.batch_sampler.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        video, video_asd, audio, audio_asd, target = batch
        if audio.shape[1] == 0:
            return
        output = self.model(video, video_asd, audio, audio_asd)
        loss = self.criterion(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self.postprocess = PostProcessor(self.args)

    def validation_step(self, batch, batch_idx):
        video, video_asd, audio, audio_asd, target = batch
        output = self.model(video, video_asd, audio, audio_asd)
        self.postprocess.update(output, target)

    def validation_epoch_end(self, outputs):
        self.postprocess.save()
        mAP, acc = self.postprocess.get_mAP()
        self.log("val_mAP", mAP)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        video, video_asd, audio, audio_asd, sid, fid_list = batch
        output = self.model(video, video_asd, audio, audio_asd)
        self.test_postprocess.update(output, sid, fid_list)

    def test_epoch_end(self, outputs):
        self.test_postprocess.save()
        self.test_postprocess.mkfile()
        pred2json(os.path.join(self.args.exp_path, 'result', 'pred.csv'), 'submit_ttm.json')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        args = self.args
        train_dataset = ImagerLoader2Task(args.img_path, args.wave_path, args.train_file, args.json_path,
                                     args.gt_path, stride=args.train_stride, transform=get_transform(True))
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            # shuffle=True,
            # batch_size=args.batch_size,
            batch_sampler=SequenceBatchSampler(train_dataset, args.batch_size),
            num_workers=args.num_workers,
            collate_fn=collate_fn_2task,
            pin_memory=False)
        return self.train_loader

    def val_dataloader(self):
        args = self.args
        val_dataset = ImagerLoader2Task(args.img_path, args.wave_path, args.val_file, args.json_path, args.gt_path,
                                   stride=args.val_stride, mode='val', transform=get_transform(False))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=False)
        return val_loader

    def test_dataloader(self):
        args = self.args
        test_dataset = TestImagerLoader2Task(args.test_data_path, args.seg_info, transform=get_transform(False))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=False)
        return test_loader
