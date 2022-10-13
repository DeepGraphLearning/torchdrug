import re
import inspect
import warnings
import functools

from decorator import decorator

import torch
from torch import nn

from torchdrug import data


def copy_args(obj, args=None, ignore=None):
    """
    Copy argument documentation from another function to fill the document of \*\*kwargs in this function.

    This class should be applied as a decorator.

    Parameters:
        obj (object): object to copy document from
        args (tuple of str, optional): arguments to copy.
            By default, it copies all argument documentation from ``obj``,
            except those already exist in the current function.
        ignore (tuple of str, optional): arguments to ignore
    """

    def wrapper(obj):
        sig = get_signature(obj)
        parameters = list(sig.parameters.values())
        if parameters[0].name == "cls" or parameters[0].name == "self":
            parameters.pop(0)
        docs = get_param_docs(obj)
        if len(docs) != len(parameters):
            raise ValueError("Fail to parse the docstring of `%s`. "
                             "Inconsistent number of parameters in signature and docstring." % obj.__name__)
        new_params = []
        new_docs = []
        param_names = {p.name for p in parameters}
        for param, doc in zip(parameters, docs):
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                for arg in from_args:
                    if arg.name in param_names:
                        continue
                    new_params.append(arg)
                    new_docs.append(from_docs[arg.name])
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                for kwarg in from_kwargs:
                    if kwarg.name in param_names:
                        continue
                    new_params.append(kwarg)
                    new_docs.append(from_docs[kwarg.name])
            else:
                new_params.append(param)
                new_docs.append(doc)

        new_sig = sig.replace(parameters=new_params)
        set_signature(obj, new_sig)
        set_param_docs(obj, new_docs)

        return obj

    from_obj = obj
    if args is not None:
        args = set(args)
    if ignore is not None:
        ignore = set(ignore)

    sig = get_signature(from_obj)
    parameters = list(sig.parameters.values())
    if parameters[0].name == "cls" or parameters[0].name == "self":
        parameters.pop(0)
    from_args = []
    from_kwargs = []
    for param in parameters:
        if (args is None or param.name in args) and (ignore is None or param.name not in ignore):
            if param.default == inspect._empty:
                from_args.append(param)
            else:
                from_kwargs.append(param)

    from_docs = get_param_docs(from_obj, as_dict=True)
    if len(from_docs) != len(parameters):
        raise ValueError("Fail to parse the docstring of `%s`. "
                         "Inconsistent number of parameters in signature and docstring." % from_obj.__name__)

    return wrapper


class cached_property(property):
    """
    Cache the property once computed.
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        result = self.func(obj)
        obj.__dict__[self.func.__name__] = result
        return result


def cached(forward, debug=False):
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

        sig = inspect.signature(forward)
        func = sig.bind(self, *args, **kwargs)
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

    return wrapper(forward)


def deprecated_alias(**alias):
    """
    Handle argument alias for a function and output deprecated warnings.
    """

    def decorate(obj):

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            for key, value in alias.items():
                if key in kwargs:
                    if value in kwargs:
                        raise TypeError("%s() got values for both `%s` and `%s`" % (obj.__name__, value, key))
                    warnings.warn("%s(): argument `%s` is deprecated in favor of `%s`" % (obj.__name__, key, value))
                    kwargs[value] = kwargs.pop(key)

            return obj(*args, **kwargs)

        sig = get_signature(obj)
        parameters = list(sig.parameters.values())
        param_docs = get_param_docs(obj, as_dict=True)
        docs = list(param_docs.values())
        alias_params = []
        alias_docs = []
        for key, value in alias.items():
            param = inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                      default=None, annotation=sig.parameters[value].annotation)
            alias_params.append(param)
            param_doc = param_docs[value]
            match = re.search(r" \(.*?\)", param_doc)
            if match:
                type_str = match.group()
            else:
                type_str = ""
            alias_docs.append("%s%s: deprecated alias of ``%s``" % (key, type_str, value))

        if parameters[-1].kind == inspect.Parameter.VAR_KEYWORD:
            new_params = parameters[:-1] + alias_params + parameters[-1:]
            new_docs = docs[:-1] + alias_docs + docs[-1:]
        else:
            new_params = parameters + alias_params
            new_docs = docs + alias_docs
        new_sig = sig.replace(parameters=new_params)
        set_signature(wrapper, new_sig)
        set_param_docs(wrapper, new_docs)

        return wrapper

    return decorate


def get_param_docs(obj, as_dict=False):
    doc = obj.__doc__ or ""

    match = re.search(r"Parameters:\n", doc)
    if not match:
        return []
    begin = match.end()
    indent = re.search(r"\s+", doc[begin:]).group()
    match = re.search(r"^(?!%s)" % indent, doc[begin:])
    if match:
        end = begin + match.start()
    else:
        end = None
    param_docs = []
    pattern = r"^%s\S.*(?:\n%s\s+\S.*)*" % (indent, indent)
    for match in re.finditer(pattern, doc[begin:end], re.MULTILINE):
        doc = match.group()
        doc = re.sub("^%s" % indent, "", doc, re.MULTILINE)  # remove indent
        param_docs.append(doc)
    if as_dict:
        param_docs = {re.search("\S+", doc).group(): doc for doc in param_docs}

    return param_docs


def set_param_docs(obj, param_docs):
    doc = obj.__doc__ or ""
    if isinstance(param_docs, dict):
        param_docs = param_docs.values()

    match = re.search(r"Parameters:\n", doc)
    if not match:
        indent = None
        for match in re.finditer(r"^(\s*)", doc):
            if indent is None or len(match.group(1)) < len(indent):
                indent = match.group(1)
        param_docs = [re.sub("^", indent, doc, re.MULTILINE) for doc in param_docs]  # add indent
        param_docs = "\n".join(param_docs)
        doc = "\n".join([doc, "%sParameters" % indent, param_docs])
    else:
        begin = match.end()
        indent = re.search(r"\s*", doc[begin:]).group()
        pattern = r"^%s\S.*(?:\n%s\s+\S.*)*(?:\n%s\S.*(?:\n%s\s+\S.*)*)*" % ((indent,) * 4)
        end = begin + re.search(pattern, doc[begin:], re.MULTILINE).end()
        param_docs = [re.sub("^", indent, doc, re.MULTILINE) for doc in param_docs]  # add indent
        param_docs = "\n".join(param_docs)
        doc = "".join([doc[:begin], param_docs, doc[end:]])
    obj.__doc__ = doc


def get_signature(obj):
    if hasattr(obj, "__signature__"):  # already overrided
        sig = obj.__signature__
    elif inspect.isclass(obj):
        sig = inspect.signature(obj.__init__)
    else:
        sig = inspect.signature(obj)

    return sig


def set_signature(obj, sig):
    doc = obj.__doc__ or ""
    match = re.search(r"^\s*\W+\(.*?\)( *-> *\W+)?", doc, re.MULTILINE)
    if not match:
        doc = "%s%s\n%s" % (obj.__name__, sig, doc)
    else:
        begin, end = match.span()
        doc = "".join([doc[:begin], obj.__name__, str(sig), doc[end:]])
    obj.__doc__ = doc
    obj.__signature__ = sig
