import torch


def collate_fn(batch):
    batch = batch[0]    # wrapper
    # in case of len 0
    new_batch = [item for item in batch if len(item["atype"]) > 0]
    batch = new_batch
    pair_flag = batch[0].get("x1") is not None

    if pair_flag:
        keys = ["atype", "btype", "edge_mask", "mask", "x0", "x1", "b0", "b1"]
        types = [torch.long] * 3 + [torch.bool] + [torch.float] * 4
        flattens = [True] * 4 + [False] * 4
    else:
        keys = ["atype", "btype", "edge_mask", "mask", "x0", "b0"]
        types = [torch.long] * 3 + [torch.bool] + [torch.float] * 2
        flattens = [True] * 4 + [False] * 2

    if batch[0].get("x_ref") is not None:
        keys.append("x_ref")
        types.append(torch.float)
        flattens.append(False)

    res = {}
    for key, _type, flatten in zip(keys, types, flattens):
        val = []
        for item in batch:
            val.append(torch.tensor(item[key], dtype=_type))
        if flatten:
            res[key] = torch.cat(val, dim=0)
        else:
            res[key] = torch.vstack(val)
    
    # special: bond index
    atom_num = 0
    bond_index = []
    for item in batch:
        bond_index.append(torch.tensor(item["bond_index"], dtype=torch.long) + atom_num)
        atom_num += len(item["atype"])
    res["bond_index"] = torch.cat(bond_index, dim=1)    # (2, E)

    abid = []
    i = 0
    for item in batch:
        abid.extend([i] * len(item["atype"]))
        i += 1
    res["abid"] = torch.tensor(abid, dtype=torch.long)  # (N,)

    # # environment
    # res["env"] = batch[0]["env"]

    return res
