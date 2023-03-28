import re
import types
import inspect
from collections import defaultdict
from contextlib import contextmanager

from decorator import decorator


class _MetaContainer(object):
    """
    Meta container that maintains meta types about members.

    The meta type of each member is tracked when a member is assigned.
    We use a context manager to define the meta types for a bunch of assignment.

    The meta types are stored as a dict in ``instance.meta_dict``,
    where keys are member names and values are meta types.

    >>> class MyClass(_MetaContainer):
    >>>     ...

    >>> instance = MyClass()
    >>> with instance.context("important"):
    >>>     instance.value = 1
    >>> assert instance.meta_dict["value"] == "important"

    Members assigned with :meth:`context(None) <context>` or without a context won't be tracked.

    >>> instance.random = 0
    >>> assert "random" not in instance.meta_dict

    You can also restrict available meta types by defining a set :attr:`_meta_types` in the derived class.

    .. note::

        Meta container also supports auto inference of meta types.
        This can be enabled by setting :attr:`enable_auto_context` to ``True`` in the derived class.

        Once auto inference is on, any member without an explicit context will be recognized through their name prefix.
        For example, ``instance.node_value`` will be recognized as ``node`` if ``node`` is defined in ``meta_types``.

        This may make code hard to maintain. Use with caution.
    """

    _meta_types = set()
    enable_auto_context = False

    def __init__(self, meta_dict=None, **kwargs):
        if meta_dict is None:
            meta_dict = {}
        else:
            meta_dict = meta_dict.copy()

        self._setattr("_meta_contexts", set())
        self._setattr("meta_dict", meta_dict)
        for k, v in kwargs.items():
            self._setattr(k, v)

    @contextmanager
    def context(self, type):
        """
        Context manager for assigning members with a specific meta type.
        """
        if type is not None and self._meta_types and type not in self._meta_types:
            raise ValueError("Expect context type in %s, but got `%s`" % (self._meta_types, type))
        self._meta_contexts.add(type)
        yield
        self._meta_contexts.remove(type)

    def __setattr__(self, key, value):
        if hasattr(self, "meta_dict"):
            types = self._meta_contexts
            if not types and self.enable_auto_context:
                for type in self._meta_types:
                    if key.startswith(type):
                        types.append(type)
                if len(types) > 1:
                    raise ValueError("Auto context found multiple contexts for key `%s`. "
                                     "If this is desired, set `enable_auto_context` to False "
                                     "and manually specify the context. " % key)
            if types:
                self.meta_dict[key] = types.copy()
        self._setattr(key, value)

    def __delattr__(self, key):
        if hasattr(self, "meta_dict") and key in self.meta_dict:
            del self.meta_dict[key]
            del self.data_dict[key]
        super(_MetaContainer, self).__delattr__(self, key)

    def _setattr(self, key, value):
        return super(_MetaContainer, self).__setattr__(key, value)

    @property
    def data_dict(self):
        """A dict that maps tracked names to members."""
        return {k: getattr(self, k) for k in self.meta_dict}

    def data_by_meta(self, include=None, exclude=None):
        """
        Return members based on the specific meta types.

        Parameters:
            include (list of string, optional): meta types to include
            exclude (list of string, optional): meta types to exclude

        Returns:
            (dict, dict): data member dict and meta type dict
        """
        if include is None and exclude is None:
            return self.data_dict, self.meta_dict

        include = self._standarize_type(include)
        exclude = self._standarize_type(exclude)
        types = include or set().union(*self.meta_dict.values())
        types = types - exclude
        data_dict = {}
        meta_dict = {}
        for k, v in self.meta_dict.items():
            if v.issubset(types):
                data_dict[k] = getattr(self, k)
                meta_dict[k] = v
        return data_dict, meta_dict

    def _standarize_type(self, types):
        if types is None:
            types = set()
        elif isinstance(types, str):
            types = {types}
        else:
            types = set(types)
        return types


class Tree(defaultdict):

    def __init__(self):
        super(Tree, self).__init__(Tree)

    def flatten(self, prefix=None, result=None):
        if prefix is None:
            prefix = ""
        else:
            prefix = prefix + "."
        if result is None:
            result = {}
        for k, v in self.items():
            if isinstance(v, Tree):
                v.flatten(prefix + k, result)
            else:
                result[prefix + k] = v
        return result


class Registry(object):
    """
    Registry class for managing all call-by-name access to objects.

    Typical scenarios:

    1. Create a model according to a string.

    >>> gcn = R.search("GCN")(128, [128])

    2. Register a customize hook to the package.

    >>> @R.register("features.atom.my_feature")
    >>> def my_featurizer(atom):
    >>>     ...
    >>>
    >>> data.Molecule.from_smiles("C1=CC=CC=C1", atom_feature="my_feature")
    """

    table = Tree()

    def __new__(cls):
        raise ValueError("Registry shouldn't be instantiated.")

    @classmethod
    def register(cls, name):
        """
        Register an object with a canonical name. Hierarchical names are separated by ``.``.
        """

        def wrapper(obj):
            entry = cls.table
            keys = name.split(".")
            for key in keys[:-1]:
                entry = entry[key]
            if keys[-1] in entry:
                raise KeyError("`%s` has already been registered by %s" % (name, entry[keys[-1]]))

            entry[keys[-1]] = obj
            obj._registry_key = name

            return obj

        return wrapper

    @classmethod
    def get(cls, name):
        """
        Get an object with a canonical name. Hierarchical names are separated by ``.``.
        """
        entry = cls.table
        keys = name.split(".")
        for i, key in enumerate(keys):
            if key not in entry:
                raise KeyError("Can't find `%s` in `%s`" % (key, ".".join(keys[:i])))
            entry = entry[key]
        return entry

    @classmethod
    def search(cls, name):
        """
        Search an object with the given name. The name doesn't need to be canonical.

        For example, we can search ``GCN`` and get the object of ``models.GCN``.
        """
        keys = []
        pattern = re.compile(r"\b%s\b" % name)
        for k, v in cls.table.flatten().items():
            if pattern.search(k):
                keys.append(k)
                value = v
        if len(keys) == 0:
            raise KeyError("Can't find any registered key containing `%s`" % name)
        if len(keys) > 1:
            keys = ["`%s`" % key for key in keys]
            raise KeyError("Ambiguous key `%s`. Found %s" % (name, ", ".join(keys)))
        return value


class _Configurable(type):

    def config_dict(self):

        def unroll_config_dict(obj):
            if isinstance(type(obj), _Configurable):
                obj = obj.config_dict()
            elif isinstance(obj, (str, bytes)):
                return obj
            elif isinstance(obj, dict):
                return type(obj)({k: unroll_config_dict(v) for k, v in obj.items()})
            elif isinstance(obj, (list, tuple)):
                return type(obj)(unroll_config_dict(x) for x in obj)
            return obj

        cls = getattr(self, "_registry_key", self.__class__.__name__)
        config = {"class": cls}
        for k, v in self._config.items():
            config[k] = unroll_config_dict(v)
        return config

    @classmethod
    def load_config_dict(cls, config):
        if cls == _Configurable:
            real_cls = Registry.search(config["class"])
            custom_load_func = real_cls.load_config_dict.__func__ != cls.load_config_dict.__func__
            if custom_load_func:
                return real_cls.load_config_dict(config)
            cls = real_cls
        elif getattr(cls, "_registry_key", cls.__name__) != config["class"]:
            raise ValueError("Expect config class to be `%s`, but found `%s`" % (cls.__name__, config["class"]))

        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict) and "class" in v:
                v = _Configurable.load_config_dict(v)
            elif isinstance(v, list):
                v = [_Configurable.load_config_dict(_v) 
                        if isinstance(_v, dict) and "class" in _v else _v 
                            for _v in v]
            if k != "class":
                new_config[k] = v

        return cls(**new_config)

    def __new__(typ, *args, **kwargs):

        cls = type.__new__(typ, *args, **kwargs)

        @decorator
        def wrapper(init, self, *args, **kwargs):
            sig = inspect.signature(init)
            func = sig.bind(self, *args, **kwargs)
            func.apply_defaults()
            config = {}
            keys = list(sig.parameters.keys())
            for k, v in zip(keys[1:], func.args[1:]): # exclude self
                config[k] = v
            config.update(func.kwargs)
            for k in getattr(self, "_ignore_args", {}):
                config.pop(k)
            self._config = dict(config)
            return init(self, *args, **kwargs)

        def get_function(method):
            if isinstance(method, types.MethodType):
                return method.__func__
            return method

        if isinstance(cls.__init__, types.FunctionType):
            cls.__init__ = wrapper(cls.__init__)
            custom_load_func = hasattr(cls, "load_config_dict") and \
                          get_function(cls.load_config_dict) != get_function(typ.load_config_dict)
            custom_config_func = hasattr(cls, "config_dict") and \
                                 get_function(cls.config_dict) != get_function(typ.config_dict)
            if not custom_load_func:
                cls.load_config_dict = _Configurable.load_config_dict
            if not custom_config_func:
                cls.config_dict = _Configurable.config_dict

        return cls


class Configurable(metaclass=_Configurable):
    """
    Class for load/save configuration.
    It will automatically record every argument passed to the ``__init__`` function.

    This class is inspired by :meth:`state_dict()` in PyTorch, but designed for hyperparameters.

    Inherit this class to construct a configurable class.

    >>> class MyClass(nn.Module, core.Configurable):

    Note :class:`Configurable` only applies to the current class rather than any derived class.
    For example, the following definition only records the arguments of ``MyClass``.

    >>> class DerivedClass(MyClass):

    In order to record the arguments of ``DerivedClass``, explicitly specify the inheritance.

    >>> class DerivedClass(MyClass, core.Configurable):

    To get the configuration of an instance, use :meth:`config_dict()`,
    which returns a dict of argument names and values.
    If an argument is also an instance of :class:`Configurable`, it will be recursively expanded in the dict.
    The configuration dict can be passed to :meth:`load_config_dict()` to create a copy of the instance.

    For classes already registered in :class:`Registry`,
    they can be directly created from the :class:`Configurable` class.
    This is convenient for building models from configuration files.

    >>> config = models.GCN(128, [128]).config_dict()
    >>> gcn = Configurable.load_config_dict(config)
    """
    pass


def make_configurable(cls, module=None, ignore_args=()):
    """
    Make a configurable class out of an existing class.
    The configurable class will automatically record every argument passed to its ``__init__`` function.

    Parameters:
        cls (type): input class
        module (str, optional): bind the output class to this module.
            By default, bind to the original module of the input class.
        ignore_args (set of str, optional): arguments to ignore in the ``__init__`` function
    """
    ignore_args = set(ignore_args)
    module = module or cls.__module__
    Metaclass = type(cls)
    if issubclass(Metaclass, _Configurable): # already a configurable class
        return cls
    if Metaclass != type: # already have a meta class
        MetaClass = type(_Configurable.__name__, (Metaclass, _Configurable), {})
    else:
        MetaClass = _Configurable
    return MetaClass(cls.__name__, (cls,), {"_ignore_args": ignore_args, "__module__": module})
