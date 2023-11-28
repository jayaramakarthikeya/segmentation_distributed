
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
import os , glob
import pickle as pkl
import utils
import matplotlib.pyplot as plt
from utils import pallete


class ADE20KDataset(Dataset):

    def __init__(self,root, split, mean, std, base_size=None, augment=True, val=False,
                crop_size=None, scale=True, flip=False, rotate=False, blur=False) -> None:
        super().__init__()

        self.DATASET_PATH = '../ADE20K_2021_17_01/'
        index_file = 'index_ade20k.pkl'
        self.num_classes = 150
        self.pallete = pallete.ADE20K_palette
        with open('{}/{}'.format(self.DATASET_PATH, index_file), 'rb') as f:
            self.index_ade20k = pkl.load(f)
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)


    def _set_files(self):
        if self.split == "validation":
            self.image_label_dir = os.path.join(self.DATASET_PATH, 'images/ADE', self.split)
            self.files = [path for path in glob(self.image_label_dir + '/**/*.jpg',recursive=True)]
        elif self.split == "training":
            self.files = [os.path.join(self.root,self.index_ade20k['folder'][i],self.index_ade20k['filename'][i]) 
                          for i in range(len(self.index_ade20k['filename']))]
        else: raise ValueError(f"Invalid split name {self.split}")

    def _val_augmentation(self, image, label):
        if self.crop_size:
            h, w = label.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)

            # Center Crop
            h, w = label.shape
            start_h = (h - self.crop_size )// 2
            start_w = (w - self.crop_size )// 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        return image, label

    def _augmentation(self, image, label):

        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller 
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    
        h, w, _ = image.shape
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
            # Cropping 
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return image, label
    
    def _load_data(self, index):
        image_path = self.files[index]
        label_path = image_path.replace('.jpg', '_seg.png')
        image = cv2.imread(image_path)[:,:,::-1]
        label = cv2.imread(label_path)[:,:,::-1]
        return image, label
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        image, label = self._load_data(index)
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        label = label.permute(2,0,1)
        image = Image.fromarray(image)
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
    

if __name__ == "__main__":

    MEAN = [0.48897059, 0.46548275, 0.4294]
    STD = [0.22861765, 0.22948039, 0.24054667]
    root = '..'
    dataset = ADE20KDataset(root=root,split='training',mean=MEAN,std=STD,base_size=550,crop_size=321)
    print("Dataset Length: ",len(dataset))
    i = 0
    for data in dataset:
        if i == 10:
            break
        print(data[0].shape,data[1].shape)
        f = plt.figure(figsize=(24,10))
        plt.subplot(1,2,1)
        plt.imshow(data[0].permute(1,2,0).numpy())
        plt.subplot(1,2,2)
        plt.imshow(data[1].permute(1,2,0).numpy())
        plt.axis('off')  # Hide axes
        plt.show()
        i += 1

