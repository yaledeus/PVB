#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_mean, scatter_sum, scatter_std
import torch.nn.functional as F

from utils.nn_utils import stable_norm

from .radial_basis import RadialBasis

from .tools import _unit_edges_from_block_edges

import pdb

class GETEncoder(nn.Module):
    def __init__(self, hidden_size, radial_size, n_channel, n_rbf,
                 edge_size=16, n_layers=3, dropout=0.1,
                 z_requires_grad=True, use_ffn=True, pre_norm=False) -> None:
        super().__init__()

        self.encoder = GET(
            hidden_size, radial_size, n_channel, n_rbf,
            edge_size, n_layers, dropout=dropout,
            z_requires_grad=z_requires_grad,
            use_ffn=use_ffn, pre_norm=pre_norm
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, pred_Z = self.encoder(H, Z, block_id, batch_id, edges, edge_attr)
        # block_repr = scatter_mean(H, block_id, dim=0)           # [Nb, hidden]
        block_repr = scatter_sum(H, block_id, dim=0)           # [Nb, hidden]
        # block_repr = F.normalize(block_repr, dim=-1)
        # graph_repr = scatter_mean(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        # graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, pred_Z


class GET(nn.Module):
    '''Equivariant Adaptive Block Transformer'''

    def __init__(self, d_hidden, d_radial, n_channel, n_rbf, d_edge=0, n_layers=4,
                 act_fn=nn.SiLU(), dropout=0.1, z_requires_grad=True, use_ffn=True, pre_norm=False):
        super().__init__()
        '''
        :param d_hidden: Number of hidden features
        :param d_radial: Number of features for calculating geometric relations
        :param n_channel: Number of channels of coordinates of each unit
        :param d_edge: Number of features for the edge features
        :param n_layers: Number of layer
        :param act_fn: Non-linearity
        :param dropout: probability of dropout
        '''


        self.n_layers = n_layers

        self.use_ffn = use_ffn

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}', EquivariantSubmodule(
                GETLayer(d_hidden, d_radial, n_channel, n_rbf, d_edge, act_fn),
                d_hidden=d_hidden, n_channel=n_channel, pre_norm=pre_norm
                )
            )
            # self.add_module(f'layernorm0_{i}', EquivariantLayerNorm(d_hidden, n_channel))
            if use_ffn:
                self.add_module(f'ffn_{i}', EquivariantSubmodule(
                    EquivariantFFN(
                        d_hidden, 4 * d_hidden, d_hidden, n_channel,
                        act_fn,  dropout,
                        z_requires_grad=z_requires_grad if i == n_layers - 1 else True
                        ),
                        d_hidden=d_hidden, n_channel=n_channel, pre_norm=pre_norm
                    )
                )
                # self.add_module(f'layernorm1_{i}', EquivariantLayerNorm(d_hidden, n_channel))

        self.pre_norm = pre_norm
        if self.pre_norm:
            self.out_norm = EquivariantLayerNorm(d_hidden, n_channel)
        # if not z_requires_grad and use_ffn:
        #     self._modules[f'layernorm1_{n_layers - 1}'].sigma.requires_grad = False
    
    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, cached_unit_edge_info=None):
        if cached_unit_edge_info is None:
            with torch.no_grad():
                cached_unit_edge_info = _unit_edges_from_block_edges(block_id, edges.T) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j

        (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = cached_unit_edge_info
            
        for i in range(self.n_layers):

            H, Z = self._modules[f'layer_{i}'](
                H, Z, block_id, batch_id, edges, edge_attr, cached_unit_edge_info
            )
            # H, Z = self._modules[f'layernorm0_{i}'](H, Z, block_id, batch_id)
            if self.use_ffn:
                H, Z = self._modules[f'ffn_{i}'](H, Z, block_id, batch_id)
                # H, Z = self._modules[f'layernorm1_{i}'](H, Z, block_id, batch_id)

        if self.pre_norm:
            H, Z = self.out_norm(H, Z, block_id, batch_id)

        return H, Z

class EquivariantSubmodule(nn.Module):

    def __init__(self, ori_module, d_hidden, n_channel, pre_norm=False):
        super(EquivariantSubmodule, self).__init__()

        self.ori_module = ori_module
        self.layernorm = EquivariantLayerNorm(d_hidden, n_channel)
        self.pre_norm = pre_norm

    def forward(self, H, Z, block_id, batch_id, *args):

        H0, Z0 = H, Z

        if self.pre_norm:
            H, Z = self.layernorm(H, Z, block_id, batch_id)

        H, Z = self.ori_module(H, Z, block_id, batch_id, *args)

        H = H + H0
        Z = Z + Z0


        if not self.pre_norm:
            H, Z = self.layernorm(H, Z, block_id, batch_id)

        return H, Z


'''
Below are the implementation of the equivariant adaptive block message passing mechanism
'''

class GETLayer(nn.Module):
    '''
    Equivariant Adaptive Block Transformer layer
    '''

    def __init__(self, d_hidden, d_radial, n_channel, n_rbf, d_edge=0,
                 act_fn=nn.SiLU(), cutoff=7.0):
        super(GETLayer, self).__init__()

        self.reci_sqrt_d = 1 / math.sqrt(d_radial)
        self.epsilon = 1e-8

        self.cutoff = cutoff
        self.n_rbf = n_rbf


        self.linear_qk = nn.Linear(d_hidden, d_radial * 2, bias=False)
        self.linear_v = nn.Linear(d_hidden, d_radial)


        self.dist_mlp = nn.Sequential(
            nn.Linear(n_rbf, d_radial, bias=False),
            act_fn
        )

        if d_edge != 0:
            self.edge_mlp = nn.Sequential(
                nn.Linear(d_edge, d_hidden),
                act_fn,
                nn.Linear(d_hidden, 1),
                act_fn
            )

        self.node_mlp = nn.Sequential(
            nn.Linear(d_hidden + d_radial, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden),
            act_fn
        )
        # self.unit_msg_mlp = nn.Sequential(
        #     nn.Linear(d_hidden + d_hidden + n_channel, d_hidden),
        #     act_fn,
        #     nn.Linear(d_hidden, d_hidden),
        #     act_fn
        # )

        if n_rbf > 1:
            self.rbf = RadialBasis(
                num_radial=n_rbf,
                cutoff=self.cutoff
            )

        self.unit_msg_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + n_rbf, d_radial),
            act_fn,
            nn.Linear(d_radial, d_radial),
            act_fn
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(1, n_channel),
            act_fn
        )
    
    def attention(self, H, Z, edges, edge_attr, cached_unit_edge_info):
        row, col = edges
        (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = cached_unit_edge_info

        # calculate attention
        H_qk = self.linear_qk(H)
        H_q, H_k = H_qk[..., 0::2][unit_row], H_qk[..., 1::2][unit_col]  # [Eu, d_radial]

        dZ = Z[unit_row] - Z[unit_col]  # [E_u, n_channel, 3]

        # D = dZ.bmm(dZ.transpose(1, 2)).view(D.shape[0], -1) # [Eu, n_channel^2]
        # D_norm = torch.norm(D + 1e-16, dim=-1, keepdim=True)
        # D = D / (1 + D_norm)
        # D = torch.norm(dZ + 1e-16, dim=-1)  # [Eu, n_channel]
        # D = stable_norm(dZ, dim=-1)  # [Eu, n_channel]
        D = torch.norm(dZ + 1e-16, dim=-1)


        if self.n_rbf > 1:
            D = self.rbf(D.squeeze(-1))


        # R = self.reci_sqrt_d * (H_q * H_k).sum(-1) + self.dist_mlp(D).squeeze()   # [Eu]

        R = (H_q * H_k * self.dist_mlp(D)).sum(-1)

        alpha = F.silu(R).unsqueeze(-1)

        # alpha = scatter_softmax(R, unit_edge_src_id).unsqueeze(-1) # [Eu, 1], unit-level attention within block-level edges

        beta = scatter_mean(R, block_edge_id) # [Eb]
        if edge_attr is not None:
            beta = beta + self.edge_mlp(edge_attr).squeeze()
        # beta = scatter_softmax(beta, row)  # [Eb], block-level edge attention
        beta = F.silu(beta)
        beta = beta[block_edge_id[unit_edge_src_start]].unsqueeze(-1)  # [Em, 1], Em = \sum_{i, j \in E} n_i

        return alpha, beta, (D, R, dZ)

    def invariant_update(self, H, alpha, beta, D, cached_unit_edge_info):
        (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = cached_unit_edge_info
        unit_agg_row = unit_row[unit_edge_src_start]

        # update invariant feature
        # H_v = self.linear_v(H)[unit_col]  # [Eu, d_radial]
        H_v = self.unit_msg_mlp(torch.cat([H[unit_row], H[unit_col], D], dim=-1))
        # H_v = self.unit_msg_mlp(torch.cat([H_v, D], dim=-1))  # [Eu, d_radial]


        H_agg = scatter_sum(alpha * H_v, unit_edge_src_id, dim=0)  # [Em, hidden_size]


        H_agg = scatter_sum(beta * H_agg, unit_agg_row, dim=0, dim_size=H.shape[0])  # [N, hidden_size]


        H_agg = self.node_mlp(torch.cat([H, H_agg], dim=-1))

        return H_agg
    
    def equivariant_update(self, Z, alpha, beta, R, dZ, cached_unit_edge_info):
        (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = cached_unit_edge_info
        unit_agg_row = unit_row[unit_edge_src_start]

        # update equivariant feature
        Z_agg = scatter_sum(
            (alpha * self.coord_mlp(R.unsqueeze(-1))).unsqueeze(-1) * dZ,
            unit_edge_src_id, dim=0)  # [Em, n_channel, 3]
        Z_agg = scatter_sum(
            beta.unsqueeze(-1) * Z_agg, unit_agg_row,
            dim=0, dim_size=Z.shape[0])  # [N, n_channel, 3]

        return Z_agg

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, cached_unit_edge_info=None):
        '''
        H: [N, hidden_size],
        Z: [N, n_channel, 3],
        block_id: [N],
        edges: [2, E], list of [n_row] and [n_col] where n_row == n_col == E, nodes from col are used to update nodes from row
        edge_attr: [E]
        cached_unit_edge_info: unit level (row, col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) calculated from block edges
        '''
        with torch.no_grad():
            if cached_unit_edge_info is None:
                cached_unit_edge_info = _unit_edges_from_block_edges(block_id, edges.T) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j


        alpha, beta, (D, R, dZ) = self.attention(H, Z, edges, edge_attr, cached_unit_edge_info)

        H = self.invariant_update(H, alpha, beta, D, cached_unit_edge_info)

        # Z = self.equivariant_update(Z, alpha, beta, R, dZ, cached_unit_edge_info)
        Z = torch.zeros_like(Z)

        return H, Z


class EquivariantFFN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_channel, act_fn=nn.SiLU(), dropout=0.1, constant=1, z_requires_grad=True) -> None:
        super().__init__()
        self.constant = constant

        self.mlp_h = nn.Sequential(
            nn.Linear(d_in * 3, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            act_fn,
            nn.Dropout(dropout)
        )

        self.mlp_z = nn.Sequential(
            nn.Linear(d_in * 3, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_channel),
            nn.Dropout(dropout)
        )

        if not z_requires_grad:
            for param in self.mlp_z.parameters():
                param.requires_grad = False
            
        self.linear_radial = nn.Linear(n_channel * n_channel, d_in, bias=False)
    
    def forward(self, H, Z, block_id, batch_id):
        '''
        :param H: [N, d_in]
        :param Z: [N, n_channel, 3]
        :param block_id: [Nb]
        '''
        radial, (Z_c, Z_o) = self._radial(Z, block_id)  # [N, n_hidden_channel], ([N, 1, 3], [N, n_channel, 3]
        H_c = scatter_mean(H, block_id, dim=0)[block_id]  # [N, d_in]
        inputs = torch.cat([H, H_c, radial], dim=-1)  # [N, d_in + d_in + d_in]

        H_update = self.mlp_h(inputs)

        # Z = Z_c + self.mlp_z(inputs).unsqueeze(-1) * Z_o
        Z_update = torch.zeros_like(Z)

        return H_update, Z_update

    def _radial(self, Z, block_id):
        Z_c = scatter_mean(Z, block_id, dim=0)  # [Nb, n_channel, 3]
        Z_c = Z_c[block_id]
        Z_o = Z - Z_c  # [N, n_channel, 3], no translation
        radial = Z_o.bmm(Z_o.transpose(1, 2))  # [N, n_channel, n_channel], no orthogonal transformation
        radial = radial.reshape(Z.shape[0], -1)  # [N, n_channel^2]
        # radial_norm = torch.norm(radial + 1e-16, dim=-1, keepdim=True)  # [N, 1]
        radial_norm = stable_norm(radial, dim=-1, keepdim=True)  # [N, 1]
        radial = radial / (self.constant + radial_norm)  # normalize for numerical stability
        radial = self.linear_radial(radial)  # [N, d_in]
        return radial, (Z_c, Z_o)


class EquivariantLayerNorm(nn.Module):
    
    def __init__(self, d_hidden, n_channel) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(d_hidden)
        sigma = torch.ones((1, n_channel, 1))
        self.sigma = nn.Parameter(sigma, requires_grad=True)
        self.eps = 1e-10

    def forward(self, H, Z, block_id, batch_id):
        with torch.no_grad():
            _, n_channel, n_axis = Z.shape
            unit_batch_id = batch_id[block_id]
            unit_axis_batch_id = unit_batch_id.unsqueeze(-1).repeat(1, n_axis).flatten()  # [N * 3]
        H = self.layernorm(H)
        # Z_c = scatter_mean(Z, unit_batch_id, dim=0)  # [bs, n_channel, 3]
        # Z_c = Z_c[unit_batch_id]  # [N, n_channel, 3]
        # Z_centered = Z - Z_c
        # var = scatter_std(
        #     Z_centered.transpose(1, 2).reshape(-1, n_channel).contiguous(),
        #     unit_axis_batch_id, dim=0)  # [bs, n_channel]
        # var = var[unit_batch_id].unsqueeze(-1)  # [N, n_channel, 1]
        # Z = Z_c + Z_centered / (var + self.eps)  * self.sigma
        # Z = Z_c + Z_centered * self.sigma
        return H, Z



if __name__ == '__main__':
    d_hidden = 64
    d_radial = 16
    n_channel = 2
    d_edge = 16
    device = torch.device('cuda:0')
    model = GET(d_hidden, d_radial, n_channel, d_edge)
    model.to(device)
    model.eval()
    
    block_id = torch.tensor([0,0,1,1,1,1,2,2,2,3,4,4,5,6,6,6,6,7,7], dtype=torch.long).to(device)
    batch_id = torch.tensor([0,0,0,0,0,1,1,1], dtype=torch.long).to(device)
    src_dst = torch.tensor([[0,1], [2,3], [1,3], [2,4], [3, 0], [3, 3], [5,7], [7,6], [5,6], [6,7]], dtype=torch.long).to(device)
    src_dst = src_dst.T
    edge_attr = torch.randn(len(src_dst[0]), d_edge).to(device)
    n_unit = block_id.shape[0]

    H = torch.randn(n_unit, d_hidden, device=device)
    Z = torch.randn(n_unit, n_channel, 3, device=device)

    H1, Z1 = model(H, Z, block_id, batch_id, src_dst, edge_attr)

    # random rotaion matrix
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q1, t1 = U.mm(V), torch.randn(3, device=device)
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q2, t2 = U.mm(V), torch.randn(3, device=device)

    unit_batch_id = batch_id[block_id]
    Z[unit_batch_id == 0] = torch.matmul(Z[unit_batch_id == 0], Q1) + t1
    Z[unit_batch_id == 1] = torch.matmul(Z[unit_batch_id == 1], Q2) + t2
    # Z = torch.matmul(Z, Q) + t

    H2, Z2 = model(H, Z, block_id, batch_id, src_dst, edge_attr)

    print(f'invariant feature: {torch.abs(H1 - H2).sum()}')
    Z1[unit_batch_id == 0] = torch.matmul(Z1[unit_batch_id == 0], Q1) + t1
    Z1[unit_batch_id == 1] = torch.matmul(Z1[unit_batch_id == 1], Q2) + t2
    print(f'equivariant feature: {torch.abs(Z1 - Z2).sum()}')