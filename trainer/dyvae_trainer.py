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


class DyVAETrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        super().__init__(model, train_loader, valid_loader, config)

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False, accumulation=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True, accumulation=True)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False, accumulation=False):
        log_type = 'valid' if val else 'train'

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            loss, (kl_loss, rec_vel_loss, rec_drf_loss) = self.model.module._train(batch, mode="pretrain")
        else:
            loss, (kl_loss, rec_vel_loss, rec_drf_loss) = self.model._train(batch, mode="pretrain")

        self.log(f'Loss/{log_type}', loss, batch_idx, val, accumulation)
        self.log(f'KL Loss/{log_type}', kl_loss, batch_idx, val, accumulation)
        self.log(f'rec. vel Loss/{log_type}', rec_vel_loss, batch_idx, val, accumulation)
        self.log(f'rec. drf Loss/{log_type}', rec_drf_loss, batch_idx, val, accumulation)

        if not val:
            # log lr
            lr = self.optimizer.param_groups[0]["lr"]
            self.log('lr', lr, batch_idx, val)

        return loss
