#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

import utils.register as R
from utils.oom_decorator import oom_decorator
from data.format import VOCAB

from .predictor import PredictorModel
from .predictor_noisy import PredictorNNModel

aas = [  # amino acids (1-letter symbol, 3-letter abbreviation)
        ('G', 'GLY'), ('A', 'ALA'), ('V', 'VAL'), ('L', 'LEU'),
        ('I', 'ILE'), ('F', 'PHE'), ('W', 'TRP'), ('Y', 'TYR'),
        ('D', 'ASP'), ('H', 'HIS'), ('N', 'ASN'), ('E', 'GLU'),
        ('K', 'LYS'), ('Q', 'GLN'), ('M', 'MET'), ('R', 'ARG'),
        ('S', 'SER'), ('T', 'THR'), ('C', 'CYS'), ('P', 'PRO') # 20 aa
        # ('U', 'SEC') # 21 aa for eukaryote
    ]

@R.register('InverseFolding')
class InverseFolding(PredictorModel):
    def __init__(self, graph_constructor: dict, encoder: dict) -> None:
        super().__init__(encoder, graph_constructor)
        # disable energy head
        for param in self.energy_ffn.parameters():
            param.requires_grad = False

        self.mask_idx = VOCAB.get_mask_idx()

        # residue type index mapping, from original index to 0~20, 0 is unk
        self.s_map = [0 for _ in range(len(VOCAB))]
        self.s_remap = [0 for _ in range(len(aas))]
        for i, (a, _) in enumerate(aas):
            original_idx = VOCAB.symbol_to_idx(a) 
            self.s_map[original_idx] = i
            self.s_remap[i] =  original_idx
        self.s_map = nn.Parameter(torch.tensor(self.s_map, dtype=torch.long), requires_grad=False)
        self.s_remap = nn.Parameter(torch.tensor(self.s_remap, dtype=torch.long), requires_grad=False)

        # binary classification head
        # self.class_head = nn.Linear(hidden_size, n_task)
        self.class_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.s_remap.shape[0])
        )

    @oom_decorator
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label=None):
        not_global = (B != self.global_block_id)
        true_B = B.clone()[not_global]
        B = torch.ones_like(B) * self.mask_idx
        B[~not_global] = self.global_block_id
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, None, return_loss=False)
        pred_seq_logits = self.class_head(return_value.block_repr)  # [bs, n_task]
        batch_ids = return_value.batch_id
        pred_seq_logits = pred_seq_logits[not_global]
        B = B.clone()[not_global]
        batch_ids = batch_ids.clone()[not_global]
        loss = F.cross_entropy(pred_seq_logits, self.s_map[true_B], reduction='none')
        loss = scatter_mean(loss, batch_ids) # [bs]
        loss = loss.sum()
        pred_seq = self.s_remap[torch.argmax(pred_seq_logits, dim=-1)]
        aar = scatter_mean((pred_seq == true_B).float(), batch_ids) # [bs]
        aar = aar.mean()

        return loss, aar, pred_seq, pred_seq_logits