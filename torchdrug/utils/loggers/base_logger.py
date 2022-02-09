import time
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch


class BaseLogger(ABC):
    def __init__(self, log_interval=100):
        self.records = defaultdict(list)
        self.epoch_id = 0
        self.batch_id = 0
        self.epoch2batch = [0]
        self.log_interval = log_interval
    
    @abstractmethod
    def log(self, record, type='train'):
        pass

    @abstractmethod
    def save_hyperparams(self, hyperparams):
        pass

    def update(self, record):
        if self.batch_id % self.log_interval == 0:
            self.log(record)
        self.batch_id += 1

        for k, v in record.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.records[k].append(v)
    
    def step(self):
        self.epoch_id += 1
        self.epoch2batch.append(self.batch_id)
        index = slice(self.epoch2batch[-2], self.epoch2batch[-1])

        averages = {}
        for k in sorted(self.records.keys()):
            averages[f"average {k}"] = np.mean(self.records[k][index])
        averages['epoch'] = self.epoch_id
        self.log(averages)
