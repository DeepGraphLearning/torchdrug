from collections import deque
from collections.abc import Mapping, Sequence

import torch

from torchdrug import data


def graph_collate(batch):
    """
    Convert any list of same nested container into a container of tensors.

    For instances of :class:`data.Graph <torchdrug.data.Graph>`, they are collated
    by :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.

    Parameters:
        batch (list): list of samples with the same nested container
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, data.Graph):
        return elem.pack(batch)
    elif isinstance(elem, Mapping):
        return {key: graph_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('Each element in list of batch should be of equal size')
        return [graph_collate(samples) for samples in zip(*batch)]

    raise TypeError("Can't collate data with type `%s`" % type(elem))


class DataLoader(torch.utils.data.DataLoader):
    """
    Extended data loader for batching graph structured data.

    See `torch.utils.data.DataLoader`_ for more details.

    .. _torch.utils.data.DataLoader:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    Parameters:
        dataset (Dataset): dataset from which to load the data
        batch_size (int, optional): how many samples per batch to load
        shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch
        sampler (Sampler, optional): sampler that draws single sample from the dataset
        batch_sampler (Sampler, optional): sampler that draws a mini-batch of data from the dataset
        num_workers (int, optional): how many subprocesses to use for data loading
        collate_fn (callable, optional): merge a list of samples into a mini-batch
        kwargs: keyword arguments for `torch.utils.data.DataLoader`_
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=graph_collate, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,
                                         **kwargs)


class DataQueue(torch.utils.data.Dataset):

    def __init__(self):
        self.queue = deque()

    def append(self, item):
        self.queue.append(item)

    def pop(self):
        self.queue.popleft()

    def __getitem__(self, index):
        return self.queue[index]

    def __len__(self):
        return len(self.deque)


class ExperienceReplay(torch.utils.data.DataLoader):

    def __init__(self, cache_size, batch_size=1, shuffle=True, **kwargs):
        super(ExperienceReplay, self).__init__(DataQueue(), batch_size, shuffle, **kwargs)
        self.cache_size = cache_size

    def update(self, items):
        for item in items:
            self.dataset.append(item)
        while len(self.dataset) > self.cache_size:
            self.dataset.pop()

    @property
    def cold(self):
        return len(self.dataset) < self.cache_size