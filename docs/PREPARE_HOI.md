# Preparation (HOI)

## Task Overview
+ PNR (Point-of-no-return Keyframe Localization)
+ OSCC (Object State Change Classification)
+ AR (Action Recognition)
+ LTA (Long-term Action Anticipation)

## Directory Tree

```bash
EgoT2
├── ...
├── HOI
│   └── ...
├── data
│   ├── fho
│   │   ├── annotations
│   │   ├── pos_clips
│   │   ├── neg_clips
│   │   └── test_clips
│   └── lta
│       ├── annotations
│       └── clips
└── pretrained_models
    ├── ts_pnr.pth
    ├── ts_oscc.pth
    ├── recognition_ego4d_slowfast8x8.ckpt
    ├── recognition_kinetics_slowfast8x8.ckpt
    └── lta_slowfast_trf.ckpt
```

## PNR & OSCC
Follow the instructions [here](https://github.com/EGO4D/hands-and-objects/tree/main/state-change-localization-classification/i3d-resnet50) to process data.

(1) First, download all the required videos to a directory (this corresponds to `/datasets01/ego4d_track2/v1/full_scale` in the provided config files). Remember to modify DATA.VIDEO_DIR_PATH to your data path. 

(2) Create three data dirs to store positive clips (clips with state change), negative clips (clips without state change) and test clips. This corresponds to `../data/fho/pos_clips`, `../data/fho/neg_clips` and `../data/fho/test_clips`  in the provided config files.

(3) Download all annotations json files under `../data/pnr/annotations`.

## AR & LTA
Follow the instructions [here](https://github.com/EGO4D/forecasting/blob/main/LONG_TERM_ANTICIPATION.md) to process data.

(1) Download all videos and downsample them to 320p, save them to `../data/lta/clips`. Remember to modify `DATA.PATH_PREFIX` if you use other directory.

(2) Download all annotations json files under `../data/lta/annotations`.

