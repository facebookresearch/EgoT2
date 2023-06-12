#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from .build import MODEL_REGISTRY, build_model  # noqa
from .video_model_builder import ResNet, SlowFast  # noqa

from .sta_models import (
    ShortTermAnticipationResNet,
    ShortTermAnticipationSlowFast,
)  # noqa
from .lta_models import (
    ForecastingEncoderDecoder,
)  # noqa

# if get circular import error, comment these 3 lines
# from .lta_models_transfer import *
# from .lta_models_lta_transfer import TaskFusionMFTransformerLTA4Task
# from .lta_models_seqdecoder import ForecastingEncoderSeqDecoder