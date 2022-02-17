import pprint
import logging

from torchdrug.core import Registry as R
from torchdrug.utils import pretty


class LoggerBase(object):
    """
    Base class for loggers.

    Any custom logger should be derived from this class.
    """

    def log(self, record, step_id, category="train/batch"):
        raise NotImplementedError

    def log_config(self, config_dict):
        raise NotImplementedError


@R.register("core.ConsoleLogger")
class ConsoleLogger(LoggerBase):
    """
    Implementation of the original logger in TorchDrug conforming to the new logging architecture.
    It inherits from the BaseLogger class and implements the log and save_hyperparams methods.
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

    def log_config(self, config_dict):
        self.logger.warning(pprint.pformat(config_dict))


@R.register("core.WandbLogger")
class WandbLogger(ConsoleLogger):
    """
    Implementation of the W&B logger in TorchDrug.
    It inherits from the BaseLogger class and implements the log and save_hyperparams methods.

    Parameters:
        project (str, optional): Name of the wandb project
        name (str, optional): Name of the wandb run
        dir (str, optional): Directory to save the wandb run
    """

    def __init__(self, project=None, name=None, dir=None, **kwargs):
        super(WandbLogger, self).__init__()
        try:
            import wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Wandb is not found. Please install it with `pip install wandb`")

        self.run = wandb.init(project=project, name=name, dir=dir, reinit=True, **kwargs)

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