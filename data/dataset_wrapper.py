from typing import Callable
from tqdm import tqdm

import numpy as np
import torch
import sympy


class MixDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, *datasets, collate_fn: Callable = None, reweigh=False, max_batches=20000) -> None:
        super().__init__()
        self.datasets = datasets
        self.cum_len = []
        self.total_len = 0
        self.max_batches = max_batches
        for dataset in datasets:
            if not reweigh:
                self.total_len += len(dataset)
            else:
                self.total_len += min(max_batches, len(dataset))
            self.cum_len.append(self.total_len)
        self.collate_fn = self.datasets[0].collate_fn if collate_fn is None else collate_fn

    def update_epoch(self):
        for dataset in self.datasets:
            if hasattr(dataset, 'update_epoch'):
                dataset.update_epoch()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                return self.datasets[i].__getitem__(idx - last_cum_len)
            last_cum_len = cum_len
        return None  # this is not possible


class DynamicBatchWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, complexity, ubound_per_batch, same_origin=False) -> None:
        super().__init__()
        self.dataset = dataset
        self.indexes = list(range(len(dataset)))
        self.index_dict = self.dataset.get_index_dict()
        self.complexity = complexity
        self.eval_func = sympy.lambdify('n', sympy.simplify(complexity))
        self.ubound_per_batch = ubound_per_batch
        self.total_size = None
        self.batch_indexes = []
        self.same_origin = same_origin
        self._form_batch()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError(f"'DynamicBatchWrapper'(or '{type(self.dataset)}') object has no attribute '{attr}'")

    def update_epoch(self):
        if hasattr(self.dataset, 'update_epoch'):
            self.dataset.update_epoch()
        self._form_batch()

    ########## overload with your criterion ##########
    def _form_batch(self):
        last_batch_indexes = self.batch_indexes
        self.batch_indexes = []

        if not self.same_origin:
            np.random.shuffle(self.indexes)
            cur_complexity = 0
            batch = []

            for i in tqdm(self.indexes):
                item_len = self.eval_func(self.dataset.get_len(i))
                if item_len > self.ubound_per_batch:
                    continue
                cur_complexity += item_len
                if cur_complexity > self.ubound_per_batch:
                    self.batch_indexes.append(batch)
                    batch = []
                    cur_complexity = item_len
                batch.append(i)
            if len(batch) > 0:
                self.batch_indexes.append(batch)
        else:
            for key in self.index_dict:
                np.random.shuffle(self.index_dict[key])
            grouped_index = list(self.index_dict.values())
            np.random.shuffle(grouped_index)
            self.indexes = [idx for group in grouped_index for idx in group]
            cur_complexity = 0
            batch = []

            for i in tqdm(self.indexes):
                item_len = self.eval_func(self.dataset.get_len(i))
                if item_len > self.ubound_per_batch:
                    continue
                cur_complexity += item_len
                if cur_complexity > self.ubound_per_batch or \
                        (len(batch) > 0 and self.dataset.get_origin(i) != self.dataset.get_origin(batch[-1])):
                    self.batch_indexes.append(batch)
                    batch = []
                    cur_complexity = item_len
                batch.append(i)
            if len(batch) > 0:
                self.batch_indexes.append(batch)

        if self.total_size is None:
            self.total_size = len(self.batch_indexes)
        else:
            # control the lengths of the dataset, otherwise the dataloader will raise error
            if len(self.batch_indexes) < self.total_size:
                num_add = self.total_size - len(self.batch_indexes)
                self.batch_indexes += last_batch_indexes[:num_add]
            else:
                self.batch_indexes = self.batch_indexes[:self.total_size]

    def __len__(self):
        return len(self.batch_indexes)

    def __getitem__(self, idx):
        return [self.dataset[i] for i in self.batch_indexes[idx]]

    def collate_fn(self, batched_batch):
        batch = []
        for minibatch in batched_batch:
            batch.extend(minibatch)
        return self.dataset.collate_fn(batch)
