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
from model.vit import vit_train

import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")



#params
data_dir = '..'
batch_size = 16
crop_size = 128
base_size = 256
scale = True
augment = True

num_epochs = 10

def main(config, model):

    torch.autograd.set_detect_anomaly(True)

    train_dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split='training',\
                                  crop_size=crop_size,base_size=base_size,scale=scale,augment=augment)
    
    val_dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split='validation',
                                      crop_size=crop_size,base_size=base_size,scale=scale,augment=augment)

    
    if model == "PSPNet":
        model = PSPNet(num_classes=train_dataloader.dataset.num_classes) 

    elif model == "UNet":
        model = UNet(num_classes=train_dataloader.dataset.num_classes)

    elif model == "DeepLab":
        model = DeepLab(num_classes=train_dataloader.dataset.num_classes)
    
    elif model == "Transformer":

        vit_train(train_dataloader, val_dataloader)

        #print("Mean IOU: {} and Pixel Acc = {}".format(mean_iou, pixAcc))
    
    else:
        print("invalid model")


    if model != "Transformer":
        gpu_trainer = SingleGPUTrainer(config=config, model=model, train_loader=train_dataloader,
                                val_loader=val_dataloader)
        

        gpu_trainer.train()
   
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-m', '--model', default=None, type=str,
                           help='Specifies which model')
    args = parser.parse_args()

    config = json.load(open(args.config))

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.model)