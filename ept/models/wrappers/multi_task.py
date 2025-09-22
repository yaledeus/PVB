#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

import utils.register as R
from utils.oom_decorator import oom_decorator
from .multi_binary_classifier import MultiBinaryClassifier
from .predictor import PredictorModel
from .predictor_noisy_norm import PredictorNNNModel


@R.register('MultiTaskModel')
class MultiTaskModel(PredictorModel):
    def __init__(self, encoder: dict, graph_constructor: dict, n_binary_task: int, noise_type: str, agg_type: str,
                 sigma_begin=0.01, sigma_end=10, n_noise_level=50, rot_sigma_begin=0.01, rot_sigma_end=10,
                 rot_n_noise_level=50, denoise_loss_scale=1.0):
        super().__init__(encoder, graph_constructor)
        self.predictor = PredictorNNNModel(encoder, graph_constructor, noise_type, agg_type,
                                           sigma_begin, sigma_end, n_noise_level, rot_sigma_begin,
                                           rot_sigma_end, rot_n_noise_level, denoise_loss_scale)
        self.classifier = MultiBinaryClassifier(n_binary_task, graph_constructor, encoder)

    @oom_decorator
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label=None):
        pred_ret = self.predictor(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label)
        cls_ret = self.classifier(Z, B, A, atom_positions, block_lengths, lengths, segment_ids) # [bs, ntask]

        return pred_ret, cls_ret
