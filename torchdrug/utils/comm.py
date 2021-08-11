import os
import multiprocessing
from collections import defaultdict

import torch
from torch import distributed as dist


cpu_group = None
gpu_group = None


def get_rank():
    """
    Get the rank of this process in distributed processes.

    Return 0 for single process case.
    """
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    """
    Get the total number of distributed processes.

    Return 1 for single process case.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def get_group(device):
    """
    Get the process group corresponding to the given device.

    Parameters:
        device (torch.device): query device
    """
    group = cpu_group if device.type == "cpu" else gpu_group
    if group is None:
        raise ValueError("%s group is not initialized. Use comm.init_process_group() to initialize it"
                         % device.type.upper())
    return group


def init_process_group(backend, init_method=None, **kwargs):
    """
    Initialize CPU and/or GPU process groups.

    Parameters:
        backend (str): Communication backend. Use ``nccl`` for GPUs and ``gloo`` for CPUs.
        init_method (str, optional): URL specifying how to initialize the process group
    """
    global cpu_group
    global gpu_group

    dist.init_process_group(backend, init_method, **kwargs)
    gpu_group = dist.group.WORLD
    if backend == "nccl":
        cpu_group = dist.new_group(backend="gloo")
    else:
        cpu_group = gpu_group


def get_cpu_count():
    """
    Get the number of CPUs on this node.
    """
    return multiprocessing.cpu_count()


def synchronize():
    """
    Synchronize among all distributed processes.
    """
    if get_world_size() > 1:
        dist.barrier()


def _recursive_read(obj):
    values = defaultdict(list)
    sizes = defaultdict(list)
    if isinstance(obj, torch.Tensor):
        values[obj.dtype] += [obj.flatten()]
        sizes[obj.dtype] += [torch.tensor([obj.numel()], device=obj.device)]
    elif isinstance(obj, dict):
        for v in obj.values():
            child_values, child_sizes = _recursive_read(v)
            for k, v in child_values.items():
                values[k] += v
            for k, v in child_sizes.items():
                sizes[k] += v
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for v in obj:
            child_values, child_sizes = _recursive_read(v)
            for k, v in child_values.items():
                values[k] += v
            for k, v in child_sizes.items():
                sizes[k] += v
    else:
        raise ValueError("Unknown type `%s`" % type(obj))
    return values, sizes


def _recursive_write(obj, values, sizes=None):
    if isinstance(obj, torch.Tensor):
        if sizes is None:
            size = torch.tensor([obj.numel()], device=obj.device)
        else:
            s = sizes[obj.dtype]
            size, s = s.split([1, len(s) - 1])
            sizes[obj.dtype] = s
        v = values[obj.dtype]
        new_obj, v = v.split([size, v.shape[-1] - size], dim=-1)
        # compatible with reduce / stack / cat
        new_obj = new_obj.view(new_obj.shape[:-1] + (-1,) + obj.shape[1:])
        values[obj.dtype] = v
        return new_obj, values
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_obj[k], values = _recursive_write(v, values, sizes)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        new_obj = []
        for v in obj:
            new_v, values = _recursive_write(v, values, sizes)
            new_obj.append(new_v)
    else:
        raise ValueError("Unknown type `%s`" % type(obj))
    return new_obj, values


def reduce(obj, op="sum", dst=None):
    """
    Reduce any nested container of tensors.

    Parameters:
        obj (Object): any container object. Can be nested list, tuple or dict.
        op (str, optional): element-wise reduction operator.
            Available operators are ``sum``, ``mean``, ``min``, ``max``, ``product``.
        dst (int, optional): rank of destination worker. If not specified, broadcast the result to all workers.

    Examples::

        >>> # assume 4 workers
        >>> rank = comm.get_rank()
        >>> x = torch.rand(5)
        >>> obj = {"polynomial": x ** rank}
        >>> obj = comm.reduce(obj)
        >>> assert torch.allclose(obj["polynomial"], x ** 3 + x ** 2 + x + 1)
    """
    values = _recursive_read(obj)[0]
    values = {k: torch.cat(v) for k, v in values.items()}

    is_mean = op == "mean"
    if is_mean:
        op = "sum"
    op = getattr(dist.ReduceOp, op.upper())

    reduced = {}
    for k, v in values.items():
        dtype = v.dtype
        # NCCL can't solve bool. Cast them to byte
        if dtype == torch.bool:
            v = v.byte()
        group = get_group(v.device)
        if dst is None:
            dist.all_reduce(v, op=op, group=group)
        else:
            dist.reduce(v, op=op, dst=dst, group=group)
        if is_mean:
            v = v / get_world_size()
        reduced[k] = v.type(dtype)

    return _recursive_write(obj, reduced)[0]


def stack(obj, dst=None):
    """
    Stack any nested container of tensors. The new dimension will be added at the 0-th axis.

    Parameters:
        obj (Object): any container object. Can be nested list, tuple or dict.
        dst (int, optional): rank of destination worker. If not specified, broadcast the result to all workers.

    Examples::

        >>> # assume 4 workers
        >>> rank = comm.get_rank()
        >>> x = torch.rand(5)
        >>> obj = {"exponent": x ** rank}
        >>> obj = comm.stack(obj)
        >>> truth = torch.stack([torch.ones_like(x), x, x ** 2, x ** 3]
        >>> assert torch.allclose(obj["exponent"], truth))
    """
    values = _recursive_read(obj)[0]
    values = {k: torch.cat(v) for k, v in values.items()}

    stacked = {}
    for k, v in values.items():
        dtype = v.dtype
        # NCCL can't solve bool. Cast them to byte
        if dtype == torch.bool:
            dtype = torch.uint8
        s = torch.zeros(get_world_size(), *v.shape, dtype=dtype, device=v.device)
        s[get_rank()] = v
        group = get_group(s.device)
        if dst is None:
            dist.all_reduce(s, op=dist.ReduceOp.SUM, group=group)
        else:
            dist.reduce(s, op=dist.ReduceOp.SUM, dst=dst, group=group)
        stacked[k] = s.type(v.dtype)

    return _recursive_write(obj, stacked)[0]


def cat(obj, dst=None):
    """
    Concatenate any nested container of tensors along the 0-th axis.

    Parameters:
        obj (Object): any container object. Can be nested list, tuple or dict.
        dst (int, optional): rank of destination worker. If not specified, broadcast the result to all workers.

    Examples::

        >>> # assume 4 workers
        >>> rank = comm.get_rank()
        >>> rng = torch.arange(10)
        >>> obj = {"range": rng[rank * (rank + 1) // 2: (rank + 1) * (rank + 2) // 2]}
        >>> obj = comm.cat(obj)
        >>> assert torch.allclose(obj["range"], rng)
    """
    values, sizes = _recursive_read(obj)
    sizes = {k: torch.cat(v) for k, v in sizes.items()}

    sizes = stack(sizes)
    cated = {}
    for k, value in values.items():
        size = sizes[k].t().flatten() # sizes[k]: (num_worker, num_obj)
        dtype = value[0].dtype
        # NCCL can't solve bool. Cast them to byte
        if dtype == torch.bool:
            dtype = torch.uint8
        s = torch.zeros(size.sum(), dtype=dtype, device=value[0].device)
        obj_id = get_rank()
        world_size = get_world_size()
        offset = size[:obj_id].sum()
        for v in value:
            assert offset + v.numel() <= len(s)
            s[offset: offset + v.numel()] = v
            offset += size[obj_id: obj_id + world_size].sum()
            obj_id += world_size
        group = get_group(s.device)
        if dst is None:
            dist.all_reduce(s, op=dist.ReduceOp.SUM, group=group)
        else:
            dist.reduce(s, op=dist.ReduceOp.SUM, dst=dst, group=group)
        cated[k] = s.type(value[0].dtype)
    sizes = {k: v.sum(dim=0) for k, v in sizes.items()}

    return _recursive_write(obj, cated, sizes)[0]