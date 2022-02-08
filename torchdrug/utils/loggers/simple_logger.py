import logging

from torchdrug.utils import pretty
from torchdrug.utils.loggers.base_logger import BaseLogger


class SimpleLogger(BaseLogger):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def log(self, record):
        self.logger.warning(pretty.separator)

        for k in sorted(record.keys()):
            self.logger.warning("%s: %g" % (k, record[k]))
