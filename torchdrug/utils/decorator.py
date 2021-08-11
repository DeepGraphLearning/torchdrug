import inspect

from decorator import decorator

import torch
from torch import nn

from torchdrug import data


class cached_property(property):
    """
    Cache the property once computed.
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, obj, cls):
        result = self.func(obj)
        obj.__dict__[self.func.__name__] = result
        return result


def cached(func, debug=False):
    """
    Cache the result of last function call.
    """

    @decorator
    def wrapper(forward, self, *args, **kwargs):

        def equal(x, y):
            if isinstance(x, nn.Parameter):
                x = x.data
            if isinstance(y, nn.Parameter):
                y = y.data
            if type(x) != type(y):
                return False
            if isinstance(x, torch.Tensor):
                return x.shape == y.shape and (x == y).all()
            elif isinstance(x, data.Graph):
                if x.num_node != y.num_node or x.num_edge != y.num_edge or x.num_relation != y.num_relation:
                    return False
                edge_feature = getattr(x, "edge_feature", torch.tensor(0, device=x.device))
                y_edge_feature = getattr(y, "edge_feature", torch.tensor(0, device=y.device))
                if edge_feature.shape != y_edge_feature.shape:
                    return False
                return (x.edge_list == y.edge_list).all() and (x.edge_weight == y.edge_weight).all() \
                       and (edge_feature == y_edge_feature).all()
            else:
                return x == y

        if self.training:
            return forward(self, *args, **kwargs)

        func = inspect.signature(forward)
        func = func.bind(self, *args, **kwargs)
        func.apply_defaults()
        arguments = func.arguments.copy()
        arguments.pop(next(iter(arguments.keys())))

        if hasattr(self, "_forward_cache"):
            hit = True
            message = []
            for k, v in arguments.items():
                if not equal(self._forward_cache[k], v):
                    hit = False
                    message.append("%s: miss" % k)
                    break
                message.append("%s: hit" % k)
            if debug:
                print("[cache] %s" % ", ".join(message))
        else:
            hit = False
            if debug:
                print("[cache] cold start")
        if hit:
            return self._forward_cache["result"]
        else:
            self._forward_cache = {}

        for k, v in arguments.items():
            if isinstance(v, torch.Tensor) or isinstance(v, data.Graph):
                v = v.detach()
            self._forward_cache[k] = v
        result = forward(self, *args, **kwargs)
        self._forward_cache["result"] = result
        return result

    return wrapper(func)