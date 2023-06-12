#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

function run(){
    
  NAME=$1
  CONFIG=$2
  shift 2;

  python scripts/lta/run_lta.py \
    --job_name $NAME \
    --working_directory ${WORK_DIR} \
    --cfg $CONFIG \
    ${CLUSTER_ARGS} \
    CHECKPOINT_LOAD_MODEL_HEAD False \
    MODEL.FREEZE_BACKBONE False \
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \
    $@
}

#-----------------------------------------------------------------------------------------------#
JOB_NAME=$1
CFG_NAME=$2
WORK_DIR='./logs/ar'
mkdir -p ${WORK_DIR}

CLUSTER_ARGS="--on_cluster NUM_GPUS 8"
#CLUSTER_ARGS="NUM_GPUS 2 TRAIN.BATCH_SIZE 4 TEST.BATCH_SIZE 4"

run ${JOB_NAME} ${CFG_NAME}
# SlowFast
#BACKBONE_WTS=/checkpoint/sherryxue/ego4d/lta_models/pretrained_models/long_term_anticipation/kinetics_slowfast8x8.ckpt
#run transfer-oscc \
#    configs/recognition/ts/transfer-oscc.yaml \
#    CHECKPOINT_FILE_PATH ${BACKBONE_WTS}

#-----------------------------------------------------------------------------------------------#
#                                    OTHER OPTIONS                                              #
#-----------------------------------------------------------------------------------------------#

# # MViT
# BACKBONE_WTS=$PWD/pretrained_models/long_term_anticipation/kinetics_mvit16x4.ckpt
# run mvit \
#     configs/Ego4dRecognition/MULTIMVIT_16x4.yaml \
#     DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

# # Debug locally using a smaller batch size / fewer GPUs
# CLUSTER_ARGS="NUM_GPUS 2 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 32"