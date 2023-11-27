
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import datetime
import os
from utils import helpers
from torch.utils import tensorboard
import json
import math
from base_trainer import EarlyStopping
import logging


class Trainer:
    def __init__(self,config,model,train_loader,val_loader,optimizer,lr_sheduler,loss):
        
        self.train_loader = train_loader
        self.config = config
        self.model = model
        trainable_params = filter(lambda p:p.requires_grad, self.model.parameters())
        self.optimizer = getattr(torch.optim, config['optimizer']['type'])(trainable_params, **config['optimizer']['args'])
        self.val_loader = val_loader
        self.model_type = config['architecture']['type']
        self.device = config['device_type']
        self.loss = loss
        self.logger = logging.getLogger(self.__class__.__name__)

        self.writer_mode = 'train'

         # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        torch.backends.cudnn.benchmark = True

        # MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

        # CHECKPOINTS & TENSOBOARD
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], self.config['name'], start_time)
        helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(cfg_trainer['log_dir'], self.config['name'], start_time)
        self.writer = tensorboard.SummaryWriter(writer_dir)

        #Early Stopping
        self.early_stoping = EarlyStopping(self.model,self.model_type,self.optimizer,self.config,self.checkpoint_dir,self.mnt_mode,trace_func=self.logger)

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus
    

    def train(self):
        for epoch in range(self.epochs):
            results = self._train_epoch(epoch)
            if epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)

            self.logger.info(f"Results for {epoch} epoch: ")
            for k , v in results.items():
                self.logger.info(f" {str(k)}: {v}")
            

            self.early_stoping(results[self.mnt_metric],epoch,self.model)

            if self.early_stoping.early_stop:
                self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping.counter} epochs')
                self.logger.warning('Training Stopped')
                break


        

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)