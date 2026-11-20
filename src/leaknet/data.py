"""Loading of the scene collections and the batch size scheduling sampler."""

import math
import random

import h5py
import numpy as np
from torch.utils.data import Dataset, Sampler


def load_scenes(path, n_samples_to_load=None):
    """
    Reads a scene collection into memory as a list of dicts of numpy arrays. Just eagerly loads everything, the
    collections are small enough for that (reopening the file per item was painfully slow).
    """

    with h5py.File(path, "r") as f:
        # note: the groups come back in alphabetical order (item_0, item_1, item_10, ...), not in generation order
        items = list(f.keys())[:n_samples_to_load]
        data = [{key: np.array(f[item][key]) for key in f[item]} for item in items]

    return data


# barebone dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# custom sampler to implement variable batch size scheduling
class ScheduledBatchSampler(Sampler):
    """
    batch_sampler with batch size scheduling based on a custom dict. Use as custom batch_sampler in DataLoader().
    batch schedule = {batch size : how many epochs, ... : ...}
    """

    def __init__(self, data_source, schedule, shuffle=False, drop_last=False):
        """
        Args:
            data_source (Dataset): Dataset to sample from.
            schedule (dict)      : mapping batch size to nr of epochs e.g. {128: 10, 256: 20, 512: 50}. no duplicates!
            shuffle (bool)       : whether to shuffle the dataset indices.
            drop_last (bool)     : whether to drop the last incomplete batch.
        """

        self.data_source = data_source  # this is essentially the whole dataset
        self.num_samples = len(data_source)  # length of the whole dataset
        self.indices = list(range(self.num_samples))  # base indices. (shuffled) batches of these are returned
        self.schedule = self._expand_schedule(schedule)  # schedule dict but now with each batch size repeated * n
        self.current_epoch = 0  # to keep track of the current epoch in training loop
        self.shuffle = shuffle
        self.drop_last = drop_last

    def _expand_schedule(self, schedule):
        """
        Expand the schedule to have a list of repeated batch size values. Returns a dict.
        e.g. {128 : 3, 256 : 2} -> {0 : 128, 1 : 128, 2 : 128, 3 : 256, 4 : 256}
        """

        expanded_schedule = {}
        cumulative_epochs = 0
        for bs, nr_eps in schedule.items():
            for i in range(cumulative_epochs, cumulative_epochs + nr_eps):
                # rebuilds the schedule dict. key i is the epoch nr i and value is the batch size for that epoch
                expanded_schedule[i] = bs
            cumulative_epochs += nr_eps

        return expanded_schedule

    def set_epoch(self, epoch):
        """
        update the current epoch when called in the training loop
        note: could also be replaced with something like step() to just increment the epoch by one ...
        """

        self.current_epoch = epoch

    def _get_batch_size(self):
        """
        just convenience for getting the current batch size from the expanded schedule depending on current epoch. If
        the epoch is larger than what is specified in the schedule, it just returns the last value.
        """

        # use dict.get(key) instead of dict[key] in order to be able to specify a default for when key is missing!
        return self.schedule.get(self.current_epoch, list(self.schedule.values())[-1])

    def __iter__(self):
        batch_size = self._get_batch_size()

        # shuffle before returning the generator object for the new epoch
        if self.shuffle:
            random.shuffle(self.indices)

        # determines the start indices for each batch. range(start, stop, step).
        for start_idx in range(0, len(self.indices), batch_size):
            batch = self.indices[start_idx : start_idx + batch_size]
            # skip yielding the last semi-full batch if self.drop_last is true
            if len(batch) < batch_size and self.drop_last:
                continue
            yield batch

    def __len__(self):
        """
        just returns the number of batches in the current epoch. changes by 1 depending on whether the last semi-full
        one is dropped or not.
        """

        batch_size = self._get_batch_size()
        if self.drop_last:
            return math.floor(self.num_samples / batch_size)
        else:
            return math.ceil(self.num_samples / batch_size)
