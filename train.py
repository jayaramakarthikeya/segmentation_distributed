import sys
sys.path.append('../')
import torch
import time
import os
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
from data_utils.dataloader import ADE20KDataLoader
from model.pspnet import PSPNet
from model.unet import UNet
from model.deeplabv3 import DeepLab
from model.upernet import UperNet

#params
data_dir = '..'
batch_size = 16
crop_size = 380
base_size = 400
scale = True
augment = True

num_epochs = 180

def main(config):

    torch.autograd.set_detect_anomaly(True)

    train_dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split='training',\
                                  crop_size=crop_size,base_size=base_size,scale=scale,augment=augment)
    
    val_dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split='validation',
                                      crop_size=crop_size,base_size=base_size,scale=scale,augment=augment)

    
    unet = UNet(num_classes=train_dataloader.dataset.num_classes)
    deepLab = DeepLab(num_classes=train_dataloader.dataset.num_classes)
    pspnet = PSPNet(num_classes=train_dataloader.dataset.num_classes,backbone='resnet50') 
    upernet = UperNet(num_classes=train_dataloader.dataset.num_classes,backbone='resnet50')
    models = [unet,deepLab,pspnet,upernet]

    for model in models:

        gpu_trainer = SingleGPUTrainer(config=config, model=model, train_loader=train_dataloader,
                               val_loader=val_dataloader)
    
        gpu_trainer.train()
   
    


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