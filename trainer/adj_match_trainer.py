#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch

from .abs_trainer import Trainer
from torch import nn


def disable_grad(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def enable_grad(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


class AdjMatchTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        super().__init__(model, train_loader, valid_loader, config)

    def train_step(self, batch, batch_idx):
        loss = self.share_step(batch, batch_idx, val=False, accumulation=False)
        return loss.clip(max=10.0)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True, accumulation=True)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False, accumulation=False):
        log_type = 'valid' if val else 'train'

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            loss, (adj_match_loss,) = self.model.module._train(batch)
        else:
            loss, (adj_match_loss,) = self.model._train(batch)

        self.log(f'Loss/{log_type}', loss, batch_idx, val, accumulation)
        self.log(f'Adjoint Match Loss/{log_type}', adj_match_loss, batch_idx, val, accumulation)

        if not val:
            # log lr
            lr = self.optimizer.param_groups[0]["lr"]
            self.log('lr', lr, batch_idx, val)

        return loss