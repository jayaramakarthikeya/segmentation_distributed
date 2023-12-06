

from trainer.base_trainer import BaseTrainer
import logging
import torch.nn as nn

class DPTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, val_loader,start_epoch):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.train_loader = train_loader
        self.n_gpu = self.config['n_gpu']

        self.device , self.available_gpus = self._get_available_devices(self.n_gpu)
        self.model = nn.DataParallel(model) 
        self.model.to(self.device)
        self.train_loader = train_loader

        self.num_classes = self.train_loader.dataset.num_classes
        super(DPTrainer,self).__init__(config, self.model, self.train_loader, val_loader, self.logger,self.device,self.n_gpu,self.available_gpus,start_epoch)