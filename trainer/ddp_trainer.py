from trainer.base_trainer import BaseTrainer
import logging
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import boto3
import socket


class DDPTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, val_loader,start_epoch, world_size = None, 
                 rank = None, localhost='localhost', master_port='12355',parallel_type=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.train_loader = train_loader
        self.n_gpu = self.config['n_gpu']

        #_ , self.available_gpus = self._get_available_devices(self.n_gpu)
        self.device = rank

        #model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)

        self.model = model.to(self.device)

        self.model = DDP(model, device_ids=[self.device])

        # self.model = nn.DataParallel(model) 
        # self.model.to(self.device)
        self.train_loader = train_loader
        

        self.num_classes = self.train_loader.dataset.num_classes
        super(DDPTrainer,self).__init__(config, self.model, self.train_loader, val_loader, self.logger,self.device,
                                        self.n_gpu,world_size,start_epoch,parallel_type=parallel_type)
