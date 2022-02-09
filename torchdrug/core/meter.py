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
    def __init__(self, log_interval=100, silent=False, metric_logger='simple', project=None):
        self.records = defaultdict(list)
        self.log_interval = log_interval
        self.epoch2batch = [0]
        self.time = [time.time()]
        self.epoch_id = 0
        self.batch_id = 0
        self.silent = silent
        if isinstance(metric_logger, str):
            if metric_logger == 'simple':
                self.logger = SimpleLogger(log_interval=log_interval)
            elif metric_logger == 'wandb':
                self.logger = WandbLogger(log_interval=log_interval, project=project)
        else:
            self.logger = metric_logger

    def log(self, record, type='train'):
        self.logger.log(record, type)
        
    def update(self, record):
        """
        Update with a meter record.

        Parameters:
            record (dict): any tensor metric
        """
        self.logger.update(record)

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
