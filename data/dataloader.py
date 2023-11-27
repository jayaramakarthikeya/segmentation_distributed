import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import ADE20KDataset

class ADE20KDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= 0.0):
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

        self.dataset = ADE20KDataset(**kwargs)
        self.shuffle = shuffle
        self.nbr_examples = len(self.dataset)
        if val_split: self.train_sampler, self.val_sampler = self._split_sampler(val_split)
        else: self.train_sampler, self.val_sampler = None, None

        self.dataloader_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super(ADE20KDataLoader, self).__init__(sampler=self.train_sampler, **self.dataloader_kwargs)

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
    batch_size = 32
    split = 'training'
    crop_size = 321
    base_size = 550
    scale = True
    augment = True
    dataloader = ADE20KDataLoader(data_dir=data_dir,batch_size=batch_size,split=split,crop_size=crop_size,base_size=base_size,scale=scale,augment=augment)

    for batch in dataloader:
        print(batch[0].shape)
        break
    