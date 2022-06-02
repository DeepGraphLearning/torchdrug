import time
import logging
from collections import defaultdict

import numpy as np
import torch

from torchdrug import core
from torchdrug.utils import pretty

logger = logging.getLogger(__name__)


class Meter(object):
    """
    Meter for recording metrics and training progress.

    Parameters:
        log_interval (int, optional): log every n updates
        silent (int, optional): surpress all outputs or not
        logger (core.LoggerBase, optional): log handler
    """
    def __init__(self, log_interval=100, silent=False, logger=None):
        self.records = defaultdict(list)
        self.log_interval = log_interval
        self.epoch2batch = [0]
        self.time = [time.time()]
        self.epoch_id = 0
        self.batch_id = 0
        self.silent = silent
        self.logger = logger

    def log(self, record, category="train/batch"):
        """
        Log a record.

        Parameters:
            record (dict): dict of any metric
            category (str, optional): log category.
                Available types are ``train/batch``, ``train/epoch``, ``valid/epoch`` and ``test/epoch``.
        """
        if self.silent:
            return
            
        if category.endswith("batch"):
            step_id = self.batch_id
        elif category.endswith("epoch"):
            step_id = self.epoch_id
        self.logger.log(record, step_id=step_id, category=category)

    def log_config(self, config):
        """
        Log a hyperparameter config.

        Parameters:
            config (dict): hyperparameter config
        """
        if self.silent:
            return
            
        self.logger.log_config(config)

    def update(self, record):
        """
        Update the meter with a record.

        Parameters:
            record (dict): dict of any metric
        """
        if self.batch_id % self.log_interval == 0:
            self.log(record, category="train/batch")
        self.batch_id += 1

        for k, v in record.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.records[k].append(v)

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
        index = slice(self.epoch2batch[-2], self.epoch2batch[-1])
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

        record = {}
        for k, v in self.records.items():
            record[k] = np.mean(v[index])
        self.log(record, category="train/epoch")

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
