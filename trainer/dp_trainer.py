
import torch
from trainer.base_trainer import BaseTrainer
import logging
import torch.nn as nn

class DPTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, val_loader,start_epoch, parallel_type = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.train_loader = train_loader
        self.n_gpu = self.config['n_gpu']
        #print(model.module.model_type)

        self.device , self.available_gpus = self._get_available_devices(self.n_gpu)
        
        dev_count = torch.cuda.device_count()
        print(f"Available GPUs: {dev_count}")
        self.model = nn.DataParallel(model, device_ids=list(range(dev_count))) 
        #self.model.to(self.device)
        self.model.to('cuda:0')
        self.train_loader = train_loader
        
        self.parallel_type = parallel_type

        self.num_classes = self.train_loader.dataset.num_classes
        super(DPTrainer,self).__init__(config, self.model, self.train_loader, val_loader, self.logger,self.device,self.n_gpu,self.available_gpus,start_epoch, parallel_type = self.parallel_type)
