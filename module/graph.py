import torch
from torch import nn
import numpy as np


def _topk(input, k, dim=None, largest=True):
    """
    This function allows for repeated indices in case k is out of range.
    """
    if input.shape[1] >= k:
        return torch.topk(input, k, dim=dim, largest=largest)
    else:
        sorted_values, sorted_indices = torch.sort(input, descending=largest, dim=dim)
        num_repeats = (k // input.shape[1]) + 1
        sorted_values = sorted_values.repeat(1, num_repeats)[:, :k]
        sorted_indices = sorted_indices.repeat(1, num_repeats)[:, :k]
        return sorted_values, sorted_indices


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def edge_inclusion_mask(edge_index, bond_index):
    num_nodes = max(edge_index.max(), bond_index.max()).item() + 1

    # unique encoding: src * num_nodes + dst
    edge_code = edge_index[0] * num_nodes + edge_index[1]
    bond_code = bond_index[0] * num_nodes + bond_index[1]
    
    bond_type = torch.isin(edge_code, bond_code).long()
    return bond_type


def _construct_knn_graph(X, rule_mats, k_neighbors, reverse=False, undirected=False):
    """
    :param X: (N, 3), coordinates
    :param rule_mats: list of (N, N), valid edges after each filtering
    :param k_neighbors: neighbors of each node
    """
    src_dst = torch.nonzero(sequential_and(*rule_mats))  # (Ef, 2), full possible edges represented in (src, dst)
    BIGINT = 1e10 if not reverse else 0  # assign a large distance to invalid edges
    N = X.shape[0]
    dist = X[src_dst]  # (Ef, 2, 3)
    dist = dist[:, 0] - dist[:, 1]      # (Ef, 3)
    dist = torch.norm(dist, dim=-1)     # (Ef,)
    src_dst = src_dst.transpose(0, 1)  # (2, Ef)
    # NOTE: self-loop allowed!
    # dist[src_dst[0] == src_dst[1]] += BIGINT    # rule out i2i
    dist = (torch.ones(N, N, device=dist.device) * BIGINT).index_put_(tuple([k for k in src_dst]), dist)
    # dist_neighbors: (N, topk), dst: (N, topk)
    dist_neighbors, dst = _topk(dist, k_neighbors, dim=-1, largest=reverse)  # (N, topk)
    del dist  # release memory
    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k_neighbors)
    src, dst = src.flatten(), dst.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)
    edge_index = torch.stack([dst, src])        # (2, Ef)
    if undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1) # (2, Ef), no direction
    # ensure uniqueness of each edge
    edge_index, _ = torch.unique(edge_index, dim=1, return_inverse=True)
    edge_atm = X[edge_index.transpose(0, 1)]    # (Ef, 2, 3)
    edge_vec = edge_atm[:, 0] - edge_atm[:, 1]  # (Ef, 3)
    edge_weight = torch.norm(edge_vec, dim=-1)  # (Ef,)
    return edge_index, edge_weight, edge_vec


def _construct_cutoff_graph(X, rule_mats, cutoff_lower=0.0, cutoff_upper=5.0):
    """
    :param X: (N, 3), coordinates
    :param rule_mats: list of (N, N), valid edges after each filtering
    """
    src_dst = torch.nonzero(sequential_and(*rule_mats))  # (E_valid, 2)
    # NOTE: self-loop allowed!
    # mask = src_dst[:, 0] != src_dst[:, 1]  # no loop
    # src_dst = src_dst[mask]
    
    dist = X[src_dst[:, 0]] - X[src_dst[:, 1]]  # (E_valid, 3)
    dist = torch.norm(dist, dim=-1)  # (E_valid,)
    mask = (dist >= cutoff_lower) & (dist < cutoff_upper)
    
    edge_index = src_dst[mask].transpose(0, 1)  # (2, Ef)
    edge_atm = X[edge_index.transpose(0, 1)]    # (Ef, 2, 3)
    edge_vec = edge_atm[:, 0] - edge_atm[:, 1]  # (Ef, 3)
    edge_weight = torch.norm(edge_vec, dim=-1)  # (Ef,)
    return edge_index, edge_weight, edge_vec


def hydrogen_filter(z, pos, edge_index, H_INDEX=0, cutoff=3.5):
    src, dst = edge_index
    dist_map = torch.norm(pos[src] - pos[dst], dim=1)
    filter_mask = (dist_map > cutoff) & ((z[src] == H_INDEX) | (z[dst] == H_INDEX))
    return ~filter_mask


def construct_edges(Z, X, bid, mask, bond_index, cutoff_lower=0.0, cutoff_upper=5.0, cutoff_H=3.5, k_neighbors=64):
    N = bid.shape[0]
    # same batch
    same_bid = bid.unsqueeze(-1).repeat(1, N)
    same_bid = same_bid == same_bid.transpose(0, 1)  # (N, N)
    # same monomer (pocket or ligand)
    same_loc = mask.unsqueeze(-1).repeat(1, N)
    same_loc = same_loc == same_loc.transpose(0, 1)  # (N, N)
    # src from ligand (masked as 1)
    from_lig = mask.unsqueeze(-1).repeat(1, N) == 1  # (N, N)

    # intra_edge_index, intra_edge_weight, intra_edge_vec = _construct_cutoff_graph(
    #     X,
    #     [same_bid, same_loc],
    #     cutoff_lower=cutoff_lower,
    #     cutoff_upper=cutoff_upper
    # )

    intra_edge_index, intra_edge_weight, intra_edge_vec = _construct_knn_graph(
        X,
        [same_bid, same_loc],
        k_neighbors,
        undirected=False
    )

    inter_edge_index, inter_edge_weight, inter_edge_vec = _construct_knn_graph(
        X,
        [same_bid, ~same_loc, from_lig],
        k_neighbors,
        undirected=True
    )

    edge_index = torch.cat([intra_edge_index, inter_edge_index], dim=1)     # (2, E)
    edge_weight = torch.cat([intra_edge_weight, inter_edge_weight], dim=0)  # (E,)
    edge_vec = torch.cat([intra_edge_vec, inter_edge_vec], dim=0)           # (E, 3)
    bond_type = edge_inclusion_mask(edge_index, bond_index)                 # (E,)

    ### set X-H cutoff to 3.5 Angstrom
    # H_INDEX = 0
    # hyd_filter_mask = hydrogen_filter(Z, X, edge_index, H_INDEX=H_INDEX, cutoff=cutoff_H)
    # edge_index, edge_weight, edge_vec = edge_index[:, hyd_filter_mask], edge_weight[hyd_filter_mask], edge_vec[hyd_filter_mask]

    return edge_index, edge_weight, edge_vec, bond_type


def split_edges(edge_index, edge_weight, L):
    """
    edge_index: (2, E)
    edge_weight: (E,)
    L: number of groups
    """
    E = edge_index.shape[1]
    _, sorted_indices = torch.sort(edge_weight, descending=False)
    
    # Split the sorted edges into L groups
    split_edge_indices = []

    k = int(E / L)
    
    for l in range(L):
        start = l * k
        end = min((l + 1) * k, E)
        group_edge_index = sorted_indices[start:end]
        split_edge_indices.append(group_edge_index)
    
    return split_edge_indices


if __name__ == "__main__":
    Z = torch.tensor([0, 0, 0, 0, 0, 0, 6, 6, 6, 8], dtype=torch.long)
    X = torch.randn(10, 3) * 5.0
    bid = torch.tensor([0] * 10, dtype=torch.long)
    mask = torch.tensor([0] * 6 + [1] * 4, dtype=torch.long)
    bond_index = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [6, 7], [6, 9]]).T
    edge_index, edge_weight, edge_vec, bond_type = construct_edges(Z, X, bid, mask, bond_index, k_neighbors=4)
    print(f"bond_type: {bond_type}")
    print(f"edge_index: {edge_index}")
