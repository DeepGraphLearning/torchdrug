import logging

from torchdrug.utils import pretty
from torchdrug.utils.loggers.base_logger import BaseLogger


class SimpleLogger(BaseLogger):
    def __init__(self, log_interval=100):
        super().__init__(log_interval=log_interval)
        self.logger = logging.getLogger(__name__)
    
    def save_hyperparams(self, hyperparams):
        self.logger.warning(pretty.separator)
        for k in sorted(hyperparams.keys()):
            self.logger.warning("%s: %s" % (k, hyperparams[k]))

    def log(self, record, type='train'):
        self.logger.warning(pretty.separator)

        for k in sorted(record.keys()):
            self.logger.warning("%s %s: %g" % (type, k, record[k]))
