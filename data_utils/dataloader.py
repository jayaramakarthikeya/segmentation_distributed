import sys
sys.path.append('../')
import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from data_utils.dataset import ADE20KDataset
import matplotlib.pyplot as plt
from utils import transforms as local_transforms
from utils.helpers import colorize_mask
from torchvision import transforms

class ADE20KDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= 0.0, 
                    parallel_type = None):
        self.MEAN = [0.48897059, 0.46548275, 0.4294]
        self.STD = [0.22861765, 0.22948039, 0.24054667]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate
        }

        self.parallel_type = parallel_type
        self.dataset = ADE20KDataset(**kwargs)
        self.shuffle = shuffle
        self.nbr_examples = len(self.dataset)
        if val_split: self.train_sampler, self.val_sampler = self._split_sampler(val_split)
        else: self.train_sampler, self.val_sampler = None, None

        if parallel_type == 'ddp':
            self.dataloader_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': False,#self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True, 
            'sampler' : DistributedSampler(self.dataset)
            }
        
        else:

            self.dataloader_kwargs = {
                'dataset': self.dataset,
                'batch_size': batch_size,
                'shuffle': self.shuffle,
                'num_workers': num_workers,
                'pin_memory': True
            }

        super(ADE20KDataLoader, self).__init__(**self.dataloader_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None
        
        self.shuffle = False

        split_indx = int(self.nbr_examples * split)
        np.random.seed(0)
        
        indxs = np.arange(self.nbr_examples)
        np.random.shuffle(indxs)
        train_indxs = indxs[split_indx:]
        val_indxs = indxs[:split_indx]
        self.nbr_examples = len(train_indxs)

        train_sampler = SubsetRandomSampler(train_indxs)
        val_sampler = SubsetRandomSampler(val_indxs)
        return train_sampler, val_sampler

    def get_val_loader(self):
        if self.val_sampler is None:
            return None
        #self.init_kwargs['batch_size'] = 1
        return DataLoader(sampler=self.val_sampler, **self.init_kwargs)
    

if __name__ == "__main__":
    data_dir = '..'
    batch_size = 2
    split = 'training'
    crop_size = 700
    base_size = 1000
    scale = True
    augment = True
    dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split=split,crop_size=crop_size,base_size=base_size,scale=scale,augment=augment,num_workers=4,
                    shuffle=True, flip=True, rotate=True, blur= True)
    i=0
    
    restore_transform = transforms.Compose([
            local_transforms.DeNormalize(dataloader.MEAN, dataloader.STD),
            transforms.ToPILImage()])
    

    for images,labels in dataloader:
        if i == 5:
            break
        print(images.shape,labels.shape)
        f = plt.figure(figsize=(24,10))
        plt.subplot(2,2,1)
        plt.imshow(restore_transform(images[0]))
        plt.subplot(2,2,2)
        plt.imshow(restore_transform(images[1]))
        plt.subplot(2,2,3)
        plt.imshow(colorize_mask(labels[0].numpy(),dataloader.dataset.pallete).convert('RGB'))
        plt.subplot(2,2,4)
        plt.imshow(colorize_mask(labels[1].numpy(),dataloader.dataset.pallete).convert('RGB'))
        plt.axis('off')  # Hide axes
        plt.show()
        i += 1
    