import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import time
import numpy as np
import json
import argparse
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from trainer.base_trainer import BaseTrainer
from trainer.single_gpu_train import SingleGPUTrainer
from data.dataloader import ADE20KDataLoader
from model.base_model import BaseModel

#params
data_dir = '../data'
batch_size = 32
split = 'training'
crop_size = 321
base_size = 550
scale = True
augment = True

num_epochs = 10

def main(config):

    dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split=split,\
                                  crop_size=crop_size,base_size=base_size,scale=scale,\
                                    augment=augment, val_split=0.8)
    
    train_loader, val_loader = dataloader.train_sampler, dataloader.val_sampler

    model = BaseModel() 

    trainer = SingleGPUTrainer(config=config, model=model, train_loader=train_loader, \
                               val_loader=val_loader)
    

   # for epoch in num_epochs:
    log = trainer._train_epoch(epoch=1)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config)