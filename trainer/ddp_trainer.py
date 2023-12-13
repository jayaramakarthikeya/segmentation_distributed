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
                 rank = None, localhost='localhost', master_port='12355'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.train_loader = train_loader
        self.n_gpu = self.config['n_gpu']

        #rank, world_size = self.get_rank_world_size()

        self.setup(rank, world_size)
        #self.device , self.available_gpus = self._get_available_devices(self.n_gpu)
        self.device = rank

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)

        self.model = model.to(self.device)

        self.model = DDP(model, device_ids=[self.device])

        # self.model = nn.DataParallel(model) 
        # self.model.to(self.device)
        self.train_loader = train_loader

        self.num_classes = self.train_loader.dataset.num_classes
        super(DDPTrainer,self).__init__(config, self.model, self.train_loader, val_loader, self.logger,self.device,
                                        self.n_gpu,self.available_gpus,start_epoch)

    def setup(rank, world_size, localhost='localhost', master_port='12355'):
        os.environ['MASTER_ADDR'] =  localhost
        os.environ['MASTER_PORT'] =  master_port
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def get_rank_world_size(self):
        ec2 = boto3.resource('ec2')
        instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])

        # Assuming each instance has the same number of GPUs
        gpus_per_instance = 4

        # Create a list of private IP addresses
        private_ips = sorted([instance.private_ip_address for instance in instances])

        # Set world_size
        world_size = len(private_ips) * gpus_per_instance

        # Get the rank for the current instance
        current_ip = socket.gethostbyname(socket.gethostname())
        rank = private_ips.index(current_ip) * gpus_per_instance

        return rank, world_size