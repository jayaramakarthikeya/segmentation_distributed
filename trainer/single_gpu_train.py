
from trainer.base_trainer import BaseTrainer
import logging

class SingleGPUTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, val_loader):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.train_loader = train_loader
        self.n_gpu = self.config['n_gpu']

        self.device , self.available_gpus = self._get_available_devices(self.n_gpu)
        if len(self.available_gpus) >= 1:
            model_ = model.to(self.device)
        elif self.device == 'cpu':
            model_ = model
        self.model = model
        self.train_loader = train_loader

        self.num_classes = self.train_loader.dataset.num_classes
        super(SingleGPUTrainer,self).__init__(config, self.model, self.train_loader, val_loader, self.logger,self.device,self.n_gpu,self.available_gpus)
        