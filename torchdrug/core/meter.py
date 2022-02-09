import logging
import time
from collections import defaultdict

import numpy as np
import torch
from torchdrug.utils import pretty
from torchdrug.utils.loggers import *

logger = logging.getLogger(__name__)


class Meter(object):
    """
    Meter for recording metrics and training progress.

    Parameters:
        log_interval (int, optional): log every n updates
        silent (int, optional): surpress all outputs or not
    """
    def __init__(self, log_interval=100, silent=False, logger='simple'):
        self.records = defaultdict(list)
        self.log_interval = log_interval
        self.epoch2batch = [0]
        self.time = [time.time()]
        self.epoch_id = 0
        self.batch_id = 0
        self.silent = silent
        if isinstance(logger, str):
            if logger == 'simple':
                self.logger = SimpleLogger()
            elif logger == 'wandb':
                self.logger = None
        else:
            self.logger = logger

    def update(self, record):
        """
        Update with a meter record.

        Parameters:
            record (dict): any tensor metric
        """
        self.logger.update(record)
        # if self.batch_id % self.log_interval == 0:
        #     self.logger.log(record)
        # self.batch_id += 1

        # for k, v in record.items():
        #     if isinstance(v, torch.Tensor):
        #         v = v.item()
        #     self.records[k].append(v)

    def step(self):
        """
        Step an epoch for this meter.

        Instead of manually invoking :meth:`step()`, it is suggested to use the following line

            >>> for epoch in meter(num_epoch):
            >>>     # do something
        """
        self.epoch_id += 1
        self.epoch2batch.append(self.batch_id)
        self.time.append(time.time())
        duration = self.time[-1] - self.time[-2]
        speed = (self.epoch2batch[-1] - self.epoch2batch[-2]) / duration
        if self.silent:
            return


        logger.warning("duration: %s" % pretty.time(duration))
        logger.warning("speed: %.2f batch / sec" % speed)


        eta = (self.time[-1] - self.time[self.start_epoch]) \
              / (self.epoch_id - self.start_epoch) * (self.end_epoch - self.epoch_id)
        logger.warning("ETA: %s" % pretty.time(eta))
        if torch.cuda.is_available():
            logger.warning("max GPU memory: %.1f MiB" % (torch.cuda.max_memory_allocated() / 1e6))
            torch.cuda.reset_peak_memory_stats()

        self.logger.step()


        # logger.warning(pretty.line)
        # for k in sorted(self.records.keys()):
        #     logger.warning("average %s: %g" % (k, np.mean(self.records[k][index])))

    def __call__(self, num_epoch):
        self.start_epoch = self.epoch_id
        self.end_epoch = self.start_epoch + num_epoch

        for epoch in range(self.start_epoch, self.end_epoch):
            if not self.silent:
                logger.warning(pretty.separator)
                logger.warning("Epoch %d begin" % epoch)
            yield epoch
            if not self.silent:
                logger.warning(pretty.separator)
                logger.warning("Epoch %d end" % epoch)
            self.step()
