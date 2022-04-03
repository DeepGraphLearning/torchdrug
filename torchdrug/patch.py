import os
import inspect
import importlib

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import dataset
from torch.utils import cpp_extension
from torch import distributed as dist

from torchdrug import core, data
from torchdrug.core import Registry as R


class PatchedModule(nn.Module):

    def __init__(self):
        super(PatchedModule, self).__init__()
        # TODO: these hooks are bugged.
        # self._register_state_dict_hook(PatchedModule.graph_state_dict)
        # self._register_load_state_dict_pre_hook(PatchedModule.load_graph_state_dict)

    def graph_state_dict(self, destination, prefix, local_metadata):
        local_graphs = []
        for name, param in self._buffers.items():
            if isinstance(param, data.Graph):
                local_graphs.append(name)
                destination.pop(prefix + name)
                for t_name, tensor in zip(data.Graph._tensor_names, param.to_tensors()):
                    if tensor is not None:
                        destination[prefix + name + "." + t_name] = tensor
        if local_graphs:
            local_metadata["graph"] = local_graphs
        return destination

    @classmethod
    def load_graph_state_dict(cls, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if "graph" not in local_metadata:
            return

        for name in local_metadata["graph"]:
            tensors = []
            for t_name in data.Graph._tensor_names:
                key = prefix + name + "." + t_name
                input_tensor = state_dict.get(key, None)
                tensors.append(input_tensor)
            try:
                state_dict[prefix + name] = data.Graph.from_tensors(tensors)
                print("successfully assigned %s" % (prefix + name))
            except:
                error_msgs.append("Can't construct Graph `%s` from tensors in the state dict" % key)
        return state_dict

    @property
    def device(self):
        try:
            tensor = next(self.parameters())
        except StopIteration:
            tensor = next(self.buffers())
        return tensor.device

    def register_buffer(self, name, tensor, persistent=True):
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError("ScriptModule does not support non-persistent buffers")

        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, torch.Tensor) and not isinstance(tensor, data.Graph):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(torch.Tensor, torchdrug.data.Graph or None required)"
                            .format(torch.typename(tensor), name))
        else:
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)


class PatchedDistributedDataParallel(nn.parallel.DistributedDataParallel):

    def _distributed_broadcast_coalesced(self, tensors, buffer_size, *args, **kwargs):
        new_tensors = []
        for tensor in tensors:
            # do not broadcast graphs
            # assume graphs are already init by each process
            if isinstance(tensor, torch.Tensor):
                new_tensors.append(tensor)
        if new_tensors:
            dist._broadcast_coalesced(self.process_group, new_tensors, buffer_size, *args, **kwargs)


def _get_build_directory(name, verbose):
    root_extensions_directory = os.environ.get('TORCH_EXTENSIONS_DIR')
    if root_extensions_directory is None:
        root_extensions_directory = cpp_extension.get_default_build_root()

    if verbose:
        print('Using {} as PyTorch extensions root...'.format(
            root_extensions_directory))

    build_directory = os.path.join(root_extensions_directory, name)
    if not os.path.exists(build_directory):
        if verbose:
            print('Creating extension directory {}...'.format(build_directory))
        # This is like mkdir -p, i.e. will also create parent directories.
        baton = cpp_extension.FileBaton("lock_%s" % name)
        if baton.try_acquire():
            os.makedirs(build_directory)
            baton.release()
        else:
            baton.wait()

    return build_directory


def patch(module, name, cls):
    backup = getattr(module, name)
    setattr(module, "_%s" % name, backup)
    setattr(module, name, cls)


patch(nn, "Module", PatchedModule)
patch(nn.parallel, "DistributedDataParallel", PatchedDistributedDataParallel)
patch(cpp_extension, "_get_build_directory", _get_build_directory)

Optimizer = optim.Optimizer
for name, cls in inspect.getmembers(optim):
    if inspect.isclass(cls) and issubclass(cls, Optimizer):
        cls = core.make_configurable(cls, ignore_args=("params",))
        cls = R.register("optim.%s" % name)(cls)
        patch(optim, name, cls)

Scheduler = scheduler._LRScheduler
for name, cls in inspect.getmembers(scheduler):
    if inspect.isclass(cls) and issubclass(cls, Scheduler):
        cls = core.make_configurable(cls, ignore_args=("optimizer",))
        cls = R.register("scheduler.%s" % name)(cls)
        patch(scheduler, name, cls)

Dataset = dataset.Dataset
for name, cls in inspect.getmembers(dataset):
    if inspect.isclass(cls) and issubclass(cls, Dataset):
        cls = core.make_configurable(cls)
        cls = R.register("dataset.%s" % name)(cls)
        patch(dataset, name, cls)
importlib.reload(torch.utils.data)