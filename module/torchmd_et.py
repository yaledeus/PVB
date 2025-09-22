# Borrowed from https://github.com/torchmd/torchmd-net

# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .ieconv import EfficientIEConvLayer
from .graph import split_edges
from utils.torchmd_utils import (
    MLP,
    NeighborEmbedding,
    CosineCutoff,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
    scatter,
)


def _init_linear_(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class SO3LayerNorm(nn.Module):
    def __init__(self, dim):
        super(SO3LayerNorm, self).__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.eps = 1e-5

    def forward(self, V):
        vec_norm = torch.norm(V, dim=self.dim, keepdim=True)
        vec_std = torch.std(vec_norm, dim=-1, keepdim=True) + self.eps
        V_norm = self.gamma * V / vec_std
        return V_norm


class ParamGVPFFNLayer(nn.Module):
    def __init__(self, d_hidden, d_ffn, act_fn=nn.SiLU(), d_output=None, vector_act='none', vector_norm=False, param_dim=0):
        super(ParamGVPFFNLayer, self).__init__()

        self.d_hidden = d_hidden
        self.d_ffn = d_ffn
        self.act_fn = act_fn
        self.d_output = d_hidden if d_output is None else d_output
        self.param_dim = param_dim

        self.linear_v = nn.Linear(d_hidden, d_hidden + self.d_output, bias=False)
        self.ffn_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + param_dim, d_ffn),
            act_fn,
            nn.Linear(d_ffn, d_hidden + self.d_output)
        )

        self.vector_norm = vector_norm
        if self.vector_norm:
            self.so3_norm = SO3LayerNorm(dim=1)

        self.vector_act = vector_act
        if self.vector_act == 'layernorm':
            self.vector_layernorm = nn.LayerNorm(self.d_output)

        _init_linear_(self.linear_v)
        _init_linear_(self.ffn_mlp)

    def vector_act_func(self, Vs):
        if self.vector_act == 'none':
            return Vs
        elif self.vector_act == 'sigmoid':
            return F.sigmoid(Vs)
        elif self.vector_act == 'tanh':
            return F.tanh(Vs)
        elif self.vector_act == 'layernorm':
            return self.vector_layernorm(Vs)
        elif self.vector_act == 'one':
            return torch.ones_like(Vs)

    def forward(self, H, V, p=None):
        if hasattr(self, "so3_norm"):
            V = self.so3_norm(V)
        V_proj = self.linear_v(V)
        V1, V2 = V_proj[..., :self.d_hidden], V_proj[..., self.d_hidden:]
        scalar = torch.cat([H, V1.norm(dim=-2)], dim=1) if p is None else torch.cat([H, V1.norm(dim=-2), p], dim=-1)
        scalar_out = self.ffn_mlp(scalar)
        H_out, V_update = scalar_out[..., :self.d_hidden], scalar_out[..., self.d_hidden:]
        V_out = self.vector_act_func(V_update.unsqueeze(-2)) * V2
        return H_out, V_out


class SubLayerWrapper(nn.Module):
    def __init__(self, sub_layer, d_hidden, layer_norm='pre', residual=True):
        super(SubLayerWrapper, self).__init__()
        self.sub_layer = sub_layer
        self.d_hidden = d_hidden
        self.layer_norm = layer_norm
        self.ln = nn.LayerNorm(d_hidden)
        self.residual = residual

    def forward(self, H, V, **kwargs):
        H0, V0 = H.clone(), V.clone()
        if self.layer_norm == 'pre':
            H = self.ln(H0)
        H, V = self.sub_layer(H, V, **kwargs)
        if self.residual:
            H = H + H0
            V = V + V0
        if self.layer_norm == 'post':
            H = self.ln(H)
        return H, V


class TorchMD_VQ_ET(nn.Module):
    r"""Equivariant Transformer's architecture. From
    Equivariant Transformers for Neural Network based Molecular Potentials; P. Tholke and G. de Fabritiis.
    ICLR 2022.

    This function optionally supports periodic boundary conditions with arbitrary triclinic boxes.
    For a given cutoff, :math:`r_c`, the box vectors :math:`\vec{a},\vec{b},\vec{c}` must satisfy
    certain requirements:

    .. math::

      \begin{align*}
      a_y = a_z = b_z &= 0 \\
      a_x, b_y, c_z &\geq 2 r_c \\
      a_x &\geq 2  b_x \\
      a_x &\geq 2  c_x \\
      b_y &\geq 2  c_y
      \end{align*}

    These requirements correspond to a particular rotation of the system and reduced form of the vectors, as well as the requirement that the cutoff be no larger than half the box width.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            This attribute is passed to the torch_cluster radius_graph routine keyword
            max_num_neighbors, which normally defaults to 32. Users should set this to
            higher values if they are using higher upper distance cutoffs and expect more
            than 32 neighbors per node/atom.
            (default: :obj:`32`)
        box_vecs (Tensor, optional):
            The vectors defining the periodic box.  This must have shape `(3, 3)`,
            where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
            If this is omitted, periodic boundary conditions are not applied.
            (default: :obj:`None`)
        vector_cutoff (bool, optional): Whether to apply the cutoff to the vector features. This prevents the energy from being discontinuous at the cutoff, but may hinder training.
            (default: :obj:`False`)
        check_errors (bool, optional): Whether to check for errors in the distance module.
            (default: :obj:`True`)

    """

    def __init__(
        self,
        hidden_channels=128,
        extra_channels=0,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_b=50,
        max_num_neighbors=32,
        check_errors=False,
        box_vecs=None,
        vector_cutoff=False,
        cross_attn=False,
        dtype=torch.float32
    ):
        super(TorchMD_VQ_ET, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.extra_channels = extra_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.max_b = max_b
        self.dtype = dtype

        act_class = act_class_mapping[activation]

        self.atom_embedding = nn.Embedding(self.max_z, hidden_channels - extra_channels, dtype=dtype)
        self.block_embedding = nn.Embedding(self.max_b, hidden_channels - extra_channels, dtype=dtype)
        self.bond_embedding = nn.Embedding(2, num_rbf, dtype=dtype)     # 0: no bond, 1: bond

        self.distance = OptimizedDistance(
            cutoff_lower,
            cutoff_upper,
            max_num_pairs=-max_num_neighbors,
            return_vecs=True,
            loop=True,
            box=box_vecs,
            long_edge_index=True,
            check_errors=check_errors,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels, 2 * num_rbf, cutoff_lower, cutoff_upper, self.max_z, dtype
            )
            if neighbor_embedding
            else None
        )

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = EquivariantMultiHeadAttention(
                hidden_channels,
                2 * num_rbf,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
                vector_cutoff,
                dtype,
            )
            self.attention_layers.append(attn)

        self.out_norm = nn.LayerNorm(hidden_channels, dtype=dtype)

        self.cross_attn = cross_attn
        if cross_attn:
            self.crs_ffn = MLP(2 * hidden_channels, hidden_channels, hidden_channels,
                               activation="silu", num_hidden_layers=1)

        self.reset_parameters()

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        self.block_embedding.reset_parameters()
        self.bond_embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()
        if self.cross_attn:
            self.crs_ffn.reset_parameters()

    def forward(
        self,
        z: Tensor,
        b: Tensor,
        pos: Tensor,
        batch: Tensor,
        t: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight_0: Optional[Tensor] = None,
        edge_vec_0: Optional[Tensor] = None,
        edge_weight_t: Optional[Tensor] = None,
        edge_vec_t: Optional[Tensor] = None,
        bond_type: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert (
            t is None or t.shape[1] == self.extra_channels
        ), f"Shape of vector t {t.shape[1]} did not match extra channels {self.extra_channels}"

        # transfer z in ATOM_TYPE to BIO_ATOM_TYPE
        # biomap = torch.tensor(ATOM2BIO_MAP, dtype=z.dtype, device=z.device)
        # mask = biomap[z] == -1
        # if mask.sum() != 0:
        #     print(f"invalid atom type: {z[mask]}")
        # z = biomap[z]
        # assert torch.all(z != -1), "invalid atom type encountered"

        x = self.atom_embedding(z) + self.block_embedding(b)
        mask = edge_index[0] != edge_index[1]
        bond_attr = self.bond_embedding(bond_type)

        # split_edge_indices = split_edges(edge_index=edge_index, edge_weight=edge_weight_t, L=self.num_layers)

        if self.cross_attn:
            x0 = torch.cat([x, torch.zeros_like(t)], dim=1) if self.extra_channels > 0 else x
            edge_attr_0 = self.distance_expansion(edge_weight_0)
            edge_attr_0 = torch.cat([edge_attr_0, bond_attr], dim=1)
            if self.neighbor_embedding is not None:
                x0 = self.neighbor_embedding(z, x0, edge_index, edge_weight_0, edge_attr_0)
        
        xt = torch.cat([x, t], dim=1) if self.extra_channels > 0 else x
        # edge_index: (2, E), edge_weight: (E,), edge_vec: (E, 3)
        # edge_index, edge_weight, edge_vec = self.distance(pos, batch, box)
        # This assert must be here to convince TorchScript that edge_vec is not None
        # If you remove it TorchScript will complain down below that you cannot use an Optional[Tensor]
        assert (
            edge_vec_t is not None
        ), "Distance module did not return directional information"

        # xt
        edge_attr_t = self.distance_expansion(edge_weight_t)
        edge_attr_t = torch.cat([edge_attr_t, bond_attr], dim=1)
        # vector norm
        edge_vec_t = edge_vec_t.clone()
        edge_vec_t[mask] = edge_vec_t[mask] / torch.norm(edge_vec_t[mask], dim=1).unsqueeze(1)
        
        xt = self.neighbor_embedding(z, xt, edge_index, edge_weight_t, edge_attr_t)
        # if cross attention, mix invariant features x0 and xt
        if self.cross_attn:
            xt = torch.cat([x0, xt], dim=-1)    # (n, 2d)
            xt = self.crs_ffn(xt)               # (n, d)

        vec = torch.zeros(xt.size(0), 3, xt.size(1), device=xt.device, dtype=xt.dtype)

        for attn in self.attention_layers:
            dx, dvec = attn(xt, vec, edge_index, edge_weight_t, edge_attr_t, edge_vec_t, dim_size=z.shape[0])
            xt = xt + dx
            vec = vec + dvec

        xt = self.out_norm(xt)

        return xt, vec, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper}), "
            f"dtype={self.dtype}"
        )


class EquivariantMultiHeadAttention(nn.Module):
    """Equivariant multi-head attention layer.

    :meta private:
    """

    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
        vector_cutoff=False,
        dtype=torch.float32,
    ):
        super(EquivariantMultiHeadAttention, self).__init__()
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.layernorm = nn.LayerNorm(hidden_channels, dtype=dtype)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3, dtype=dtype)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3, dtype=dtype)

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 3, bias=False, dtype=dtype
        )

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels, dtype=dtype)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3, dtype=dtype)
        self.vector_cutoff = vector_cutoff

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, dim_size=None):
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
            if self.dv_proj is not None
            else None
        )
        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            dim_size=dim_size,
        )
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        return dx, dvec

    def propagate(
        self,
        edge_index: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        vec: Tensor,
        dk: Optional[Tensor],
        dv: Optional[Tensor],
        r_ij: Tensor,
        d_ij: Tensor,
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        q_i = q.index_select(0, edge_index[1])
        k_j = k.index_select(0, edge_index[0])
        v_j = v.index_select(0, edge_index[0])
        vec_j = vec.index_select(0, edge_index[0])
        x, vec = self.message(q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij)
        return self.aggregate((x, vec), edge_index[1], dim_size=dim_size)

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        vec_j: Tensor,
        dk: Optional[Tensor],
        dv: Optional[Tensor],
        r_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        cutoff = self.cutoff(r_ij).unsqueeze(1)
        attn = self.attn_activation(attn)

        # The original ET architecture only weights the attention with the cutoff function,
        #  this causes a discontinuity in the energy at the cutoff, since the bias of the dv_proj
        #  layer might be non-zero.
        # This option makes it so that both the scalar and vector features are weighted with the cutoff.
        if self.vector_cutoff:
            v_j = v_j * cutoff.unsqueeze(2)
        else:
            attn = attn * cutoff
        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=0, dim_size=dim_size)
        vec = scatter(vec, index, dim=0, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs
