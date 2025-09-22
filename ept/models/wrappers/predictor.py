#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch_scatter import scatter_mean, scatter_sum

import utils.register as R
from data.format import VOCAB
from utils.nn_utils import std_conserve_scatter_sum

from ..graph_constructor import GraphConstructor


ReturnValue = namedtuple(
    'ReturnValue',
    ['energy', 
     'unit_repr', 'block_repr', 'graph_repr',
     'batch_id', 'block_id',
     'loss'],
    )

@R.register('PredictorModel')
class PredictorModel(nn.Module):

    # def __init__(self, model_type, hidden_size, n_channel, n_rbf,
    #              radial_size=16, edge_size=64, k_neighbors=9, n_layers=3,
    #              dropout=0.1, std=10, global_message_passing=False, use_ffn=True, pre_norm=False,
    #              atom_level=False, **kwargs) -> None:
    def __init__(self, encoder: dict, graph_constructor: dict):
        super().__init__()

        # self.model_type = model_type
        # self.hidden_size = hidden_size
        # self.n_channel = n_channel
        # self.n_rbf = n_rbf
        # self.radial_size = radial_size
        # self.edge_size = edge_size
        # self.k_neighbors = k_neighbors
        # self.n_layers = n_layers
        # self.dropout = dropout
        # self.std = std
        # self.global_message_passing = global_message_passing
        # self.use_ffn = use_ffn
        # self.atom_level = atom_level
        self.encoder_config = encoder
        self.graph_config = graph_constructor
        self.hidden_size = self.encoder_config['hidden_size']

        self.global_block_id = VOCAB.symbol_to_idx(VOCAB.GLB)

        self.graph_constructor = R.construct(graph_constructor)
        self.encoder = R.construct(encoder, z_requires_grad=False)
        
        self.energy_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1)
        )

        # z_requires_grad = False
        # if model_type == 'GET':
        #     from ..GET.get import GETEncoder
        #     self.encoder = GETEncoder(
        #         hidden_size, radial_size, n_channel,
        #         edge_size, n_layers, dropout=dropout,
        #         z_requires_grad=z_requires_grad
        #     )
        # elif model_type == 'GET_dev':
        #     from ..GET.get_dev import GETEncoder
        #     self.encoder = GETEncoder(
        #         hidden_size, radial_size, n_channel, n_rbf,
        #         edge_size, n_layers, dropout=dropout,
        #         z_requires_grad=z_requires_grad,
        #         use_ffn=use_ffn, pre_norm = pre_norm
        #     )          
        # elif model_type == 'GET_stable':
        #     from ..GET.get_stable import GETEncoder
        #     self.encoder = GETEncoder(
        #         hidden_size, radial_size, n_channel, n_rbf,
        #         edge_size, n_layers, dropout=dropout,
        #         z_requires_grad=z_requires_grad,
        #         use_ffn=use_ffn, pre_norm = pre_norm
        #     )          
        # elif model_type == 'GET_stable_equiv':
        #     from ..GET.get_stable_equiv import GETEncoder
        #     self.encoder = GETEncoder(
        #         hidden_size, radial_size, n_channel, n_rbf,
        #         edge_size, n_layers, dropout=dropout,
        #         z_requires_grad=z_requires_grad,
        #         use_ffn=use_ffn, pre_norm = pre_norm
        #     )          
        # elif model_type == 'TorchMD':
        #     from ..torchmd.encoder import TorchMDEncoder
        #     self.encoder = TorchMDEncoder(hidden_size, edge_size, n_layers, n_rbf)  
        # elif model_type == 'EGNN':
        #     from ..egnn.encoder import EGNNEncoder
        #     self.encoder = EGNNEncoder(hidden_size, edge_size, n_layers)
        # elif model_type == 'GearNet':
        #     from ..gearnet.encoder import GearNetEncoder
        #     self.encoder = GearNetEncoder(hidden_size, edge_size, n_layers)
        # else:
        #     raise NotImplementedError(f'Model type {model_type} not implemented!')

#    @torch.no_grad() # Enable the gradient of Z so that Z.grad can be obtained for certain tasks (e.g., MD17)
    def normalize(self, Z, B, block_id, batch_id):
        # centering
        center = Z[(B[block_id] == self.global_block_id)]  # [bs]
        Z = Z - center[batch_id][block_id]
        # normalize
        # Z = Z / self.std
        return Z

    
#    @torch.no_grad() # Enable the gradient of Z so that Z.grad can be obtained for certain tasks (e.g., MD17)
    def update_global_block(self, Z, B, block_id):
        is_global = B[block_id] == self.global_block_id  # [Nu]
        scatter_ids = torch.cumsum(is_global.long(), dim=0) - 1  # [Nu]
        not_global = ~is_global
        centers = scatter_mean(Z[not_global], scatter_ids[not_global], dim=0)  # [Nglobal, n_channel, 3], Nglobal = batch_size * 2
        Z = Z.clone()
        Z[is_global] = centers
        return Z, not_global
    


    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_loss=True) -> ReturnValue:
        
        # # batch_id and block_id
        # with torch.no_grad():

        #     batch_id = torch.zeros_like(segment_ids)  # [Nb]
        #     batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        #     batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

        #     block_id = torch.zeros_like(A) # [Nu]
        #     block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
        #     block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

        #     if self.atom_level:  # this is for ablation
        #         # transform blocks to single units
        #         batch_id = batch_id[block_id]  # [Nu]
        #         segment_ids = segment_ids[block_id]  # [Nu]
        #         B = B[block_id]  # [Nu]
        #         block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]

        # # Enable the gradient of Z so that Z.grad can be obtained for certain tasks (e.g., MD17)
        # batch_size = lengths.shape[0]
        # # normalize
        # Z = self.normalize(Z, B, block_id, batch_id)

        # Z, not_global = self.update_global_block(Z, B, block_id)
        
        # # embedding
        # H_0 = self.block_embedding(B, A, atom_positions, block_id)

        # # edges and edge attributes
        # intra_edges, inter_edges, global_global_edges, global_normal_edges, seq_edges = self.edge_constructor(B, batch_id, segment_ids, X=Z, block_id=block_id)

        # if self.global_message_passing:
        #     edges = torch.cat([intra_edges, inter_edges, global_global_edges, global_normal_edges], dim=1)
        #     edge_attr = torch.cat([
        #         torch.zeros_like(intra_edges[0]),
        #         torch.ones_like(inter_edges[0]),
        #         torch.ones_like(global_global_edges[0]) * 2,
        #         torch.ones_like(global_normal_edges[0]) * 3])
        # else:
        #     edges = torch.cat([intra_edges, inter_edges], dim=1)
        #     edge_attr = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])
        # edge_attr = self.edge_embedding(edge_attr)

        graph = self.graph_constructor.forward(
            unit_type=A, unit_pos=Z, num_nodes=lengths, unit_position_ids=atom_positions,
            segment_ids=segment_ids, block_type=B, block_num_units=block_lengths
        )
        
        # normalize
        Z, B = graph.unit_pos, graph.block_type
        Z = self.normalize(Z, B, graph.unit2block, graph.batch_ids)

        Z, not_global = self.update_global_block(Z, B, graph.unit2block)

        # embedding
        H_0 = graph.unit_features
        block_id = graph.unit2block
        batch_id = graph.batch_ids
        edges = graph.edges
        edge_attr = graph.edge_attr

        not_global_edge = torch.logical_and(
            B[edges[0]] != self.global_block_id,
            B[edges[1]] != self.global_block_id
        )
        edges, edge_attr = (edges.T[not_global_edge]).T, edge_attr[not_global_edge]

        # encoding
        unit_repr, block_repr, graph_repr, pred_Z = self.encoder(H_0, Z, block_id, batch_id, edges, edge_attr)

        # predict energy
        # must be sum instead of mean! mean will make the gradient (predicted noise) pretty small, and the score net will easily converge to 0
        # pred_energy = scatter_sum(self.energy_ffn(block_repr).squeeze(-1), batch_id)
        pred_energy = std_conserve_scatter_sum(self.energy_ffn(block_repr), batch_id, dim=0).squeeze(-1)

        # predict noise level
        # pred_noise_level = self.noise_level_ffn(graph_repr)  # [batch_size, n_noise_level]


        if return_loss:
            
            loss = F.mse_loss(pred_energy, label)  # [Nperturb, n_channel, 3]

        else:
            loss = None

        return ReturnValue(

            # denoising variables
            energy=pred_energy,

            # representations
            unit_repr=unit_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # loss
            loss=loss,
        )
