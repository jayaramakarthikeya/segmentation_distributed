from trainer.base_trainer import BaseTrainer
import logging
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import functools


class FSDPTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, val_loader,start_epoch, world_size = None, 
                 rank = None, localhost='localhost', master_port='12355',parallel_type=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.train_loader = train_loader
        self.n_gpu = self.config['n_gpu']

        #_ , self.available_gpus = self._get_available_devices(self.n_gpu)
        self.device = rank
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=20000
        )

        #model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
        
        self.model = model.to(self.device)

        self.model = FSDP(model, fsdp_auto_wrap_policy=my_auto_wrap_policy)
        # self.model = nn.DataParallel(model) 
        # self.model.to(self.device)
        self.train_loader = train_loader
        

        self.num_classes = self.train_loader.dataset.num_classes
        super(FSDPTrainer,self).__init__(config, self.model, self.train_loader, val_loader, self.logger,self.device,
                                        self.n_gpu,world_size,start_epoch,parallel_type=parallel_type)
