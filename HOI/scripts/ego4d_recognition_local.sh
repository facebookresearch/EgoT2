#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

mkdir -p ./logs/ar

python scripts/lta/run_lta.py --cfg configs/recognition/baseline/MULTISLOWFAST_8x8_R101.yaml \
    NUM_GPUS 1 \
    TRAIN.BATCH_SIZE 6 \
    TEST.BATCH_SIZE 64 \
    CHECKPOINT_LOAD_MODEL_HEAD False \
    MODEL.FREEZE_BACKBONE False \
    DATA.CHECKPOINT_MODULE_FILE_PATH ""\
    OUTPUT_DIR ./logs/ar
