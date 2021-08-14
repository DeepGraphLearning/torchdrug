import os
from collections.abc import Mapping, Sequence

import torch
from torch.utils import cpp_extension

from torchdrug import data
from torchdrug.utils import comm


class LazyExtensionLoader(object):

    def __init__(self, name, sources, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None,
                 extra_include_paths=None, build_directory=None, verbose=False, **kwargs):
        self.name = name
        self.sources = sources
        self.extra_cflags = extra_cflags
        self.extra_cuda_cflags = extra_cuda_cflags
        self.extra_ldflags = extra_ldflags
        self.extra_include_paths = extra_include_paths
        worker_name = "%s_%d" % (name, comm.get_rank())
        self.build_directory = build_directory or cpp_extension._get_build_directory(worker_name, verbose)
        self.verbose = verbose
        self.kwargs = kwargs

    def __getattr__(self, key):
        if "module" not in self.__dict__:
            self.module = cpp_extension.load(self.name, self.sources, self.extra_cflags, self.extra_cuda_cflags,
                                             self.extra_ldflags, self.extra_include_paths, self.build_directory,
                                             self.verbose, **self.kwargs)
        return getattr(self.module, key)


def load_extension(name, sources, **kwargs):
    """
    Load a PyTorch C++ extension just-in-time (JIT).

    This function performs lazy evaluation and is multi-process-safe.

    See `torch.utils.cpp_extension.load`_ for more details.

    .. _torch.utils.cpp_extension.load:
        https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load
    """
    return LazyExtensionLoader(name, sources, **kwargs)


def cpu(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CPU.
    """
    if hasattr(obj, "cpu"):
        return obj.cpu(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cpu(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(cpu(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


def detach(obj):
    """
    Detach tensors in any nested conatiner.
    """
    if hasattr(obj, "detach"):
        return obj.detach()
    elif isinstance(obj, Mapping):
        return type(obj)({k: detach(v) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(detach(x) for x in obj)

    raise TypeError("Can't perform detach over object type `%s`" % type(obj))


def clone(obj, *args, **kwargs):
    """
    Clone tensors in any nested conatiner.
    """
    if hasattr(obj, "clone"):
        return obj.clone(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: clone(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(clone(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't perform detach over object type `%s`" % type(obj))


def mean(obj, *args, **kwargs):
    """
    Compute mean of tensors in any nested container.
    """
    if hasattr(obj, "mean"):
        return obj.mean(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: mean(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(mean(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't perform mean over object type `%s`" % type(obj))


def cat(objs, *args, **kwargs):
    """
    Concatenate a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.cat(objs, *args, **kwargs)
    elif isinstance(obj, data.PackedGraph):
        return data.cat(objs)
    elif isinstance(obj, Mapping):
        return {k: cat([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, Sequence):
        return type(obj)(cat(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform concatenation over object type `%s`" % type(obj))


def stack(objs, *args, **kwargs):
    """
    Stack a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.stack(objs, *args, **kwargs)
    elif isinstance(obj, Mapping):
        return {k: stack([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, Sequence):
        return type(obj)(stack(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform stack over object type `%s`" % type(obj))


def sparse_coo_tensor(indices, values, size):
    """
    Construct a sparse COO tensor without index check. Much faster than `torch.sparse_coo_tensor`_.

    .. _torch.sparse_coo_tensor:
        https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html

    Parameters:
        indices (Tensor): 2D indices of shape (2, n)
        values (Tensor): values of shape (n,)
        size (list): size of the tensor
    """
    return torch_ext.sparse_coo_tensor_unsafe(indices, values, size)


path = os.path.join(os.path.dirname(__file__), "extension")

torch_ext = load_extension("torch_ext", [os.path.join(path, "torch_ext.cpp")], extra_cflags=["-Ofast"])