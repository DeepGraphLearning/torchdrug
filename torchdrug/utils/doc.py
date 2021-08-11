import re
import inspect


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

    def get_param_docs(obj):
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
            doc = re.sub("^%s" % indent, "", doc, re.MULTILINE) # remove indent
            param_docs.append(doc)

        return param_docs

    def set_param_docs(obj, param_docs):
        doc = obj.__doc__ or ""

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
        if hasattr(obj, "__signature__"): # already overrided
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
            if param.kind == inspect._VAR_POSITIONAL:
                for arg in from_args:
                    if arg.name in param_names:
                        continue
                    new_params.append(arg)
                    new_docs.append(from_docs[arg.name])
            elif param.kind == inspect._VAR_KEYWORD:
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

    from_docs = get_param_docs(from_obj)
    from_docs = {re.search("\S+", doc).group(): doc for doc in from_docs}
    if len(from_docs) != len(parameters):
        raise ValueError("Fail to parse the docstring of `%s`. "
                         "Inconsistent number of parameters in signature and docstring." % from_obj.__name__)

    return wrapper