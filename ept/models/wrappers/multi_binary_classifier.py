#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.register as R
from utils.oom_decorator import oom_decorator

from .predictor import PredictorModel
from .predictor_noisy import PredictorNNModel


@R.register('MultiBinaryClassifier')
class MultiBinaryClassifier(PredictorModel):
    def __init__(self, n_binary_task: int, graph_constructor: dict, encoder: dict) -> None:
        super().__init__(encoder, graph_constructor)
        self.n_binary_task = n_binary_task  # how many binary classification tasks?
        # disable energy head
        for param in self.energy_ffn.parameters():
            param.requires_grad = False

        # binary classification head
        # self.class_head = nn.Linear(hidden_size, n_task)
        self.class_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_binary_task)
        )

    @oom_decorator
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label=None):
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, None, return_loss=False)
        pred_class = self.class_head(return_value.graph_repr)  # [bs, n_task]
        # pred_class = torch.sigmoid(pred_class)
        return pred_class, return_value.graph_repr

@R.register('MultiBinaryClassifierNN')
class MultiBinaryClassifierNN(PredictorNNModel):
    def __init__(self, n_binary_task: int, graph_constructor: dict, encoder: dict, nn_config: dict={}) -> None:
        super().__init__(encoder, graph_constructor, **nn_config)
        self.n_binary_task = n_binary_task  # how many binary classification tasks?
        # disable energy head
        for param in self.energy_ffn.parameters():
            param.requires_grad = False

        # binary classification head
        # self.class_head = nn.Linear(hidden_size, n_task)
        self.class_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_binary_task)
        )

    @oom_decorator
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label=None):
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, None, return_loss=False)
        pred_class = self.class_head(return_value.graph_repr)  # [bs, n_task]
        # pred_class = torch.sigmoid(pred_class)
        return pred_class, return_value.loss


@R.register('MultiBinaryClassifierDisconnected')
class MultiBinaryClassifierDisconnected(PredictorModel):
    def __init__(self, n_binary_task: int, graph_constructor: dict, encoder: dict) -> None:
        super().__init__(encoder, graph_constructor)
        self.n_binary_task = n_binary_task  # how many binary classification tasks?
        # disable energy head
        for param in self.energy_ffn.parameters():
            param.requires_grad = False

        # binary classification head

        # binary classification head
        self.class_head = nn.Linear(self.hidden_size * 2, n_binary_task)

    @oom_decorator
    def forward(self, Z_r, B_r, A_r, atom_positions_r, block_lengths_r, lengths_r, segment_ids_r,
                    Z_l, B_l, A_l, atom_positions_l, block_lengths_l, lengths_l, segment_ids_l, label=None):
        return_value_r = super().forward(Z_r, B_r, A_r, atom_positions_r, block_lengths_r, lengths_r, segment_ids_r, None, return_loss=False)
        return_value_l = super().forward(Z_l, B_l, A_l, atom_positions_l, block_lengths_l, lengths_l, segment_ids_l, None, return_loss=False)
        embedding = torch.cat([return_value_r.graph_repr, return_value_l.graph_repr], dim=-1)
        pred_class = self.class_head(embedding)  # [bs, n_task]
        pred_class = torch.sigmoid(pred_class)
        return pred_class
