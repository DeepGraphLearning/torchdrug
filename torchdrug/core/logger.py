import pprint
import logging
import warnings

from torchdrug.core import Registry as R
from torchdrug.utils import pretty


class LoggerBase(object):
    """
    Base class for loggers.

    Any custom logger should be derived from this class.
    """

    def log(self, record, step_id, category="train/batch"):
        """
        Log a record.

        Parameters:
            record (dict): dict of any metric
            step_id (int): index of this log step
            category (str, optional): log category.
                Available types are ``train/batch``, ``train/epoch``, ``valid/epoch`` and ``test/epoch``.
        """
        raise NotImplementedError

    def log_config(self, config):
        """
        Log a hyperparameter config.

        Parameters:
            config (dict): hyperparameter config
        """
        raise NotImplementedError


@R.register("core.LoggingLogger")
class LoggingLogger(LoggerBase):
    """
    Log outputs with the builtin logging module of Python.

    By default, the logs will be printed to the console. To additionally log outputs to a file,
    add the following lines in the beginning of your code.

    .. code-block: python

        import logging

        format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger = logging.getLogger("")
        logger.addHandler(handler)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def log(self, record, step_id, category="train/batch"):
        if category.endswith("batch"):
            self.logger.warning(pretty.separator)
        elif category.endswith("epoch"):
            self.logger.warning(pretty.line)
        if category == "train/epoch":
            for k in sorted(record.keys()):
                self.logger.warning("average %s: %g" % (k, record[k]))
        else:
            for k in sorted(record.keys()):
                self.logger.warning("%s: %g" % (k, record[k]))

    def log_config(self, config):
        self.logger.warning(pprint.pformat(config))


@R.register("core.WandbLogger")
class WandbLogger(LoggingLogger):
    """
    Log outputs with `Weights and Biases`_ and track the experiment progress.

    Note this class also output logs with the builtin logging module.

    See `wandb.init`_ for more details.

    .. _Weights and Biases:
        https://docs.wandb.ai

    .. _wandb.init:
        https://docs.wandb.ai/ref/python/init

    Parameters:
        project (str, optional): name of the project
        name (str, optional): name of this run
        dir (str, optional): path to store meta data. Default is `./wandb`.
        kwargs: keyword arguments for `wandb.init`_
    """

    def __init__(self, project=None, name=None, dir=None, **kwargs):
        super(WandbLogger, self).__init__()
        try:
            import wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Wandb is not found. Please install it with `pip install wandb`")

        if wandb.run is not None:
            warnings.warn(
                 "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                " this run. If this is not desired, call `wandb.finish()` or `WandbLogger.finish()` before instantiating `WandbLogger`."
            )
            self.run = wandb.run
        else:
            self.run = wandb.init(project=project, name=name, dir=dir, **kwargs)

        self.run.define_metric("train/batch/*", step_metric="batch", summary="none")
        for split in ["train", "valid", "test"]:
            self.run.define_metric("%s/epoch/*" % split, step_metric="epoch")

    def log(self, record, step_id, category="train/batch"):
        super(WandbLogger, self).log(record, step_id, category)
        record = {"%s/%s" % (category, k): v for k, v in record.items()}
        step_name = category.split("/")[-1]
        record[step_name] = step_id
        self.run.log(record)

    def log_config(self, confg_dict):
        super(WandbLogger, self).log_config(confg_dict)
        self.run.config.update(confg_dict)