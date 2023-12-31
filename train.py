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
from model.hrnet import HighResolutionNet
from utils.helpers import initialize_weights, setup, cleanup
from glob import glob
from trainer.ddp_trainer import DDPTrainer
from trainer.dp_trainer import DPTrainer
import torch.multiprocessing as mp

#params
data_dir = '..'
batch_size = 16
crop_size = 380
base_size = 400
scale = True
augment = True

num_epochs = 80

def main(rank, args, config, world_size = 1):

    torch.autograd.set_detect_anomaly(True)

    setup(rank=rank, world_size=world_size)

    train_dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split='training',\
                                  crop_size=crop_size,base_size=base_size,scale=scale,augment=augment,num_workers=8,
                                  parallel_type = args.parallel)
    
    val_dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split='validation',
                                      crop_size=crop_size,base_size=base_size,scale=scale,augment=augment,num_workers=4)

    
    if args.model == "pspnet":
        print("INITIALIZING PSPNET")
        model = PSPNet(num_classes=train_dataloader.dataset.num_classes,backbone='resnet50', parallel_type=args.parallel) 

    elif args.model == "upernet":
        model = UperNet(num_classes=train_dataloader.dataset.num_classes,backbone='resnet50', parallel_type=args.parallel)
    
    else:
        print("NO MODEL CONFIGURED. USE -m FLAG")
        exit()
    
    #checkpoint_dir = './final_model'
    #checkpoint = torch.load('/home/ubuntu/segmentation_distributed/saved/PSPNet/12-06_15-09/checkpoint-epoch5.pth')
    #start_epoch = checkpoint['epoch']
    #model.load_state_dict(checkpoint['state_dict'])
    #print(checkpoint['optimizer'])
    #trainable_params = filter(lambda p:p.requires_grad, model.parameters())
    #optimizer = getattr(torch.optim, config['optimizer']['type'])(model.parameters(), **config['optimizer']['args'])

    if args.parallel == 'dp':
        print("USING DATA PARALLEL")
        gpu_trainer = DPTrainer(config=config, model=model, train_loader=train_dataloader,
                            val_loader=val_dataloader,start_epoch=None, parallel_type=args.parallel)
    
    elif args.parallel == 'ddp':
        gpu_trainer = DDPTrainer(config=config, model=model, train_loader=train_dataloader,
                            val_loader=val_dataloader,start_epoch=None, parallel_type=args.parallel, rank = rank, world_size=world_size)

    else:
    
        gpu_trainer = SingleGPUTrainer(config=config, model=model, train_loader=train_dataloader,
                            val_loader=val_dataloader,start_epoch=None)

    print("TRAINING")
    gpu_trainer.train()

    if args.parallel == 'ddp':
        cleanup()
   
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-p', '--parallel', default=None, type=str,
                           help='Which Parallel algo? (eg. dp, ddp) (default: Single GPU)')
    parser.add_argument('-m', '--model', default=None, type=str,
                           help='Which model do you want to use? Unet, Deeplab, PSP, UperNet')
    args = parser.parse_args()

    config = json.load(open(args.config))

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    if args.parallel != 'ddp':
        main(0,args, config)
    else:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(args, config, world_size), nprocs=world_size)


