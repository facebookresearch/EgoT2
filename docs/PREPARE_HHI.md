# Preparation (HHI)

## Task Overview
+ LAM (Looking-at-me)
+ TTM (Talking-to-me)
+ ASD (Active Speaker Detetcion)


## Directory Tree

```bash
EgoT2
├── ...
├── HHI
│   └── ...
├── data
│   ├── ttm
│   │   ├── video_imgs
│   │   ├── wave
│   │   ├── split
│   │   ├── json_original
│   │   ├── result_TTM
│   │   ├── seg_info.json
│   │   └── final_test_data
│   ├── lam
│   │   └── ...
│   └── asd
│       └── ...
└── pretrained_models
    ├── ts_asd.pth
    ├── ts_lam.pth
    ├── ts_ttm.pth
    └── ...
```

## LAM
> Skip this step if you plan to adopt our provided checkpoint (`pretrained_models/ts_lam.pth`) as the task-specific model in EgoT2-s.

> This step is necessary for EgoT2-g training.

Follow the instructions [here](https://github.com/EGO4D/social-interactions/tree/lam) to preprocess data and store them under `data/lam`.

## TTM
Follow the instructions [here](https://github.com/EGO4D/social-interactions/tree/ttm) to preprocess data and store them under `data/ttm`.

## ASD
> Skip this step if your task of interest is not ASD, and you plan to adopt our provided checkpoint (`pretrained_models/ts_asd.pth`) as the task-specific model in EgoT2-s.

> This step is necessary for EgoT2-g training.

Follow the instructions [here](https://github.com/zcxu-eric/Ego4d_TalkNet_ASD/) to preprocess data and store them under `data/asd`.


