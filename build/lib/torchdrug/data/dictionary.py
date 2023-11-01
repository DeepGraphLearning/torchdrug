import math
import torch

from torch_scatter import scatter_max

from torchdrug import utils


class PerfectHash(object):
    """
    Perfect hash function.

    The function can be applied to either scalar keys or vector keys.
    It takes :math:`O(n\log n)` time and :math:`O(n)` space to construct the hash table.
    It maps queries to their indexes in the original key set in :math:`O(1)` time.
    If the query is not present in the key set, it returns -1.

    The algorithm is adapted from `Storing a Sparse Table with O(1) Worst Case Access Time`_.

    .. _Storing a Sparse Table with O(1) Worst Case Access Time:
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.91.346&rep=rep1&type=pdf

    Parameters:
        keys (LongTensor): keys of shape :math:`(N,)` or :math:`(N, D)`
        weight (LongTensor, optional): weight of the level-1 hash
        bias (LongTensor, optional): bias of the level-1 hash
        sub_weights (LongTensor, optional): weight of level-2 hashes
        sub_biases (LongTensor, optional): bias of level-2 hashes
    """

    prime = 1000000000039
    max_attempt = 10
    max_input_dim = (torch.iinfo(torch.int64).max - prime) / prime

    def __init__(self, keys, weight=None, bias=None, sub_weights=None, sub_biases=None):
        if keys.ndim == 1:
            keys = keys.unsqueeze(-1)
        num_input, input_dim = keys.shape
        if weight is None:
            weight = torch.randint(0, self.prime, (1, input_dim), device=keys.device)
        if bias is None:
            bias = torch.randint(0, self.prime, (1,), device=keys.device)
        if sub_weights is None:
            sub_weights = torch.randint(0, self.prime, (num_input, input_dim), device=keys.device)
        if sub_biases is None:
            sub_biases = torch.randint(0, self.prime, (num_input,), device=keys.device)

        self.keys = keys
        self.weight = weight
        self.bias = bias
        self.sub_weights = sub_weights
        self.sub_biases = sub_biases
        self.num_input = num_input
        self.num_output = num_input
        self.input_dim = input_dim

        self._construct_hash_table()

    def _construct_hash_table(self):
        index = self.hash(self.keys)
        count = index.bincount(minlength=self.num_output)
        for i in range(self.max_attempt):
            if (count ** 2).sum() < 4 * self.num_output:
                break
            self._reset_hash()
            index = self.hash(self.keys)
            count = index.bincount(minlength=self.num_output)
        else:
            raise RuntimeError("Fail to generate a level-1 hash after %d attempts. "
                               "Are you sure the keys are unique?" % self.max_attempt)
        self.num_sub_outputs = (count ** 2).clamp(min=1)
        self.num_sub_output = self.num_sub_outputs.sum()
        self.offsets = self.num_sub_outputs.cumsum(0) - self.num_sub_outputs

        sub_index = self.sub_hash(self.keys, index)
        count = sub_index.bincount(minlength=self.num_sub_output)
        has_collision = scatter_max(count, self.second2first, dim_size=self.num_output)[0] > 1
        max_attempt = int(self.max_attempt * math.log(self.num_input) / math.log(2))
        for i in range(max_attempt):
            if not has_collision.any():
                break
            self._reset_sub_hash(has_collision)
            sub_index = self.sub_hash(self.keys, index)
            count = sub_index.bincount(minlength=self.num_sub_output)
            has_collision = scatter_max(count, self.second2first, dim_size=self.num_output)[0] > 1
        else:
            raise RuntimeError("Fail to generate level-2 hashes after %d attempts. "
                               "Are you sure the keys are unique?" % max_attempt)

        self.table = -torch.ones(self.num_sub_output, dtype=torch.long, device=self.device)
        self.table[sub_index] = torch.arange(self.num_input, device=self.device)

    def __call__(self, keys):
        """
        Get the indexes of keys in the original key set.

        Return -1 for keys that are not present in the key set.
        """
        keys = torch.as_tensor(keys, dtype=torch.long, device=self.device)
        if self.input_dim == 1 and keys.shape[-1] != 1:
            keys = keys.unsqueeze(-1)
        index = self.hash(keys)
        sub_index = self.sub_hash(keys, index)
        final_index = self.table[sub_index]
        found = final_index != -1
        found_index = final_index[found]
        equal = (keys[found] == self.keys[final_index[found]]).all(dim=-1)
        final_index[found] = torch.where(equal, found_index, -torch.ones_like(found_index))
        return final_index

    def _reset_hash(self):
        self.weight = torch.randint_like(self.weight, 0, self.prime)
        self.bias = torch.randint_like(self.bias, 0, self.prime)

    def _reset_sub_hash(self, mask=None):
        if mask is None:
            self.sub_weights = torch.randint_like(self.sub_weights, 0, self.prime)
            self.sub_biases = torch.randint_like(self.sub_biases, 0, self.prime)
        else:
            self.sub_weights[mask] = torch.randint_like(self.sub_weights[mask], 0, self.prime)
            self.sub_biases[mask] = torch.randint_like(self.sub_biases[mask], 0, self.prime)

    def hash(self, keys):
        """Apply the level-1 hash function to the keys."""
        keys = keys % self.prime
        hash = (keys * self.weight % self.prime).sum(dim=-1) + self.bias
        return hash % self.prime % self.num_output

    def sub_hash(self, keys, index):
        """
        Apply level-2 hash functions to the keys.

        Parameters:
            keys (LongTensor): query keys
            index (LongTensor): output of the level-1 hash function
        """
        keys = keys % self.prime
        weight = self.sub_weights[index]
        bias = self.sub_biases[index]
        num_outputs = self.num_sub_outputs[index]
        offsets = self.offsets[index]
        hash = (keys * weight % self.prime).sum(dim=-1) + bias
        return hash % self.prime % num_outputs + offsets

    @utils.cached_property
    def second2first(self):
        """Level-2 hash values to level-1 hash values mapping."""
        range = torch.arange(self.num_output, device=self.device)
        second2first = range.repeat_interleave(self.num_sub_outputs)
        return second2first

    @property
    def device(self):
        """Device."""
        return self.keys.device

    def cpu(self):
        """
        Return a copy of this hash function in CPU memory.

        This is a non-op if the hash function is already in CPU memory.
        """
        keys = self.keys.cpu()

        if keys is self.keys:
            return self
        else:
            return type(self)(keys, weight=self.weight.cpu(), bias=self.bias.cpu(),
                              sub_weights=self.sub_weights.cpu(), sub_biases=self.sub_biases.cpu())

    def cuda(self, *args, **kwargs):
        """
        Return a copy of this hash function in CUDA memory.

        This is a non-op if the hash function is already on the correct device.
        """
        keys = self.keys.cuda(*args, **kwargs)

        if keys is self.keys:
            return self
        else:
            return type(self)(keys, weight=self.weight.cuda(*args, **kwargs),
                              bias=self.bias.cuda(*args, **kwargs),
                              sub_weights=self.sub_weights.cuda(*args, **kwargs),
                              sub_biases=self.sub_biases.cuda(*args, **kwargs))


class Dictionary(object):
    """
    Dictionary for mapping keys to values.

    This class has the same behavior as the built-in dict, except it operates on tensors and support batching.

    Example::

        >>> keys = torch.tensor([[0, 0], [1, 1], [2, 2]])
        >>> values = torch.tensor([[0, 1], [1, 2], [2, 3]])
        >>> d = data.Dictionary(keys, values)
        >>> assert (d[[[0, 0], [2, 2]]] == values[[0, 2]]).all()
        >>> assert (d.has_key([[0, 1], [1, 2]]) == torch.tensor([False, False])).all()

    Parameters:
        keys (LongTensor): keys of shape :math:`(N,)` or :math:`(N, D)`
        values (Tensor): values of shape :math:`(N, ...)`
        hash (PerfectHash, optional): hash function for keys
    """
    def __init__(self, keys, values, hash=None):
        self.keys = keys
        self.values = values
        self.hash = hash or PerfectHash(keys)

    def __getitem__(self, keys):
        """
        Return the value for each key. Raise key error if any key is not in the dictionary.
        """
        keys = torch.as_tensor(keys, dtype=torch.long, device=self.device)
        index = self.hash(keys)
        not_found = index == -1
        if not_found.any():
            raise KeyError(keys[not_found].tolist())
        return self.values[index]

    def get(self, keys, default=None):
        """
        Return the value for each key if the key is in the dictionary, otherwise the default value is returned.

        Parameters:
            keys (LongTensor): keys of arbitrary shape
            default (int or Tensor, optional): default return value. By default, 0 is used.
        """
        keys = torch.as_tensor(keys, dtype=torch.long, device=self.device)
        if default is None:
            default = 0
        default = torch.as_tensor(default, dtype=self.values.dtype, device=self.device)
        index = self.hash(keys)
        shape = list(index.shape) + list(self.values.shape[1:])
        values = torch.ones(shape, dtype=self.values.dtype, device=self.device) * default
        found = index != -1
        values[found] = self.values[index[found]]
        return values

    def has_key(self, keys):
        """Check whether each key exists in the dictionary."""
        index = self.hash(keys)
        return index != -1

    def to_dict(self):
        """
        Return a built-in dict object of this dictionary.
        """
        keys = self.keys.tolist()
        values = self.values.tolist()
        dict = {tuple(k): tuple(v) for k, v in zip(keys, values)}
        return dict

    @property
    def device(self):
        """Device."""
        return self.keys.device

    def cpu(self):
        """
        Return a copy of this dictionary in CPU memory.

        This is a non-op if the dictionary is already in CPU memory.
        """
        keys = self.keys.cpu()

        if keys is self.keys:
            return self
        else:
            return type(self)(keys, self.values.cpu(), hash=self.hash.cpu())

    def cuda(self, *args, **kwargs):
        """
        Return a copy of this dictionary in CUDA memory.

        This is a non-op if the dictionary is already in CUDA memory.
        """
        keys = self.keys.cuda(*args, **kwargs)

        if keys is self.keys:
            return self
        else:
            return type(self)(keys, self.values.cuda(*args, **kwargs), hash=self.hash.cuda(*args, **kwargs))