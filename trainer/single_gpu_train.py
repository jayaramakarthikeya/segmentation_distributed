

import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from trainer.base_trainer import BaseTrainer
import logging

class SingleGPUTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, val_loader):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.train_loader = train_loader
        self.n_gpu = self.config['n_gpu']

        self.num_classes = self.train_loader.dataset.num_classes
        super(SingleGPUTrainer,self).__init__(config, model, train_loader, val_loader, self.logger)
        self.device , self.available_gpus = self._get_available_devices(self.n_gpu)
        if len(self.available_gpus) >= 1:
            self.model = model.to(self.device)
        elif self.device == 'cpu':
            self.model = model
        self.train_loader = train_loader