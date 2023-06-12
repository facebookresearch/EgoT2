#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

# from .short_term_anticipation import Ego4dShortTermAnticipation
from .long_term_anticipation import Ego4dRecognition, Ego4dLongTermAnticipation
from .long_term_anticipation_auxtask import Ego4dRecognitionwithAuxTask
from .long_term_anticipation_lta_auxtask import Ego4dLongTermAnticipationwithAuxTask