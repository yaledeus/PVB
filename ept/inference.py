import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.mmap_dataset import MMAPDataset
import json
import sys
import os


def to_device(data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, 'to'):
            data = data.to(device)
        return data


def ept_predict(mmap_dir, gpu=-1):
    cwd = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(cwd)
    task = "DockedPDBBind"
    # load model
    b_ckpt = os.path.join(cwd, "ckpt/docked_pdbbind_scaffold_best.ckpt")
    print(f'Using checkpoint {b_ckpt}')
    model = torch.load(b_ckpt, map_location='cpu')
    device = torch.device('cpu' if gpu == -1 else f'cuda:{gpu}')
    model.to(device)
    model.eval()

    # load data
    test_set = MMAPDataset(mmap_dir)
    batch_size, num_workers = 32, 4
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             num_workers=num_workers,
                             collate_fn=test_set.collate_fn)
    
    # save path
    save_path = os.path.join(mmap_dir, '_results.jsonl')

    fout = open(save_path, 'w')

    idx = 0

    post_trans = lambda x: x.energy

    for batch in tqdm(test_loader):
        with torch.no_grad():
            # move data
            batch = to_device(batch, device)
            results = model(
                Z=batch['X'], B=batch['B'], A=batch['A'],
                atom_positions=batch['atom_positions'],
                block_lengths=batch['block_lengths'],
                lengths=batch['lengths'],
                segment_ids=batch['segment_ids'],
                label=batch['label'])
            results = post_trans(results)
            labels = batch['label']
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().tolist()

            for pred, label in zip(results.detach().cpu().tolist(), labels):
                out_dict = {
                        'pred': pred,
                        'task': task,
                        'gt': label
                    }

                fout.write(json.dumps(out_dict) + '\n')
                idx += 1
    
    fout.close()


if __name__ == "__main__":
    mmap_dir = sys.argv[1]
    gpu = sys.argv[2] if len(sys.argv) == 3 else -1
    ept_predict(mmap_dir, gpu)