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
from utils.helpers import initialize_weights
from glob import glob

#params
data_dir = '..'
batch_size = 16
crop_size = 380
base_size = 400
scale = True
augment = True

num_epochs = 80

def main(config):

    torch.autograd.set_detect_anomaly(True)

    train_dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split='training',\
                                  crop_size=crop_size,base_size=base_size,scale=scale,augment=augment,num_workers=8)
    
    val_dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split='validation',
                                      crop_size=crop_size,base_size=base_size,scale=scale,augment=augment,num_workers=4)

    
    unet = UNet(num_classes=train_dataloader.dataset.num_classes)
    deepLab = DeepLab(num_classes=train_dataloader.dataset.num_classes)
    pspnet = PSPNet(num_classes=train_dataloader.dataset.num_classes,backbone='resnet50') 
    upernet = UperNet(num_classes=train_dataloader.dataset.num_classes,backbone='resnet50')
    
    model = pspnet
    #checkpoint_dir = './final_model'
    #checkpoint = torch.load('/home/ubuntu/segmentation_distributed/saved/DeepLab/12-06_04-22/checkpoint-epoch76.pth')
    #start_epoch = checkpoint['epoch']
    #model.load_state_dict(checkpoint['state_dict'])
    #print(checkpoint['optimizer'])
    #trainable_params = filter(lambda p:p.requires_grad, model.parameters())
    #optimizer = getattr(torch.optim, config['optimizer']['type'])(model.parameters(), **config['optimizer']['args'])
    
    gpu_trainer = SingleGPUTrainer(config=config, model=model, train_loader=train_dataloader,
                            val_loader=val_dataloader,start_epoch=None)

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