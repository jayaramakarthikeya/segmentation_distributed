import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl
import utils

# Load index with global information about ADE20K
DATASET_PATH = '../ADE20K_2021_17_01/'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

print("File loaded, description of the attributes:")
print('--------------------------------------------')
for attribute_name, desc in index_ade20k['description'].items():
    print('* {}: {}'.format(attribute_name, desc))
print('--------------------------------------------\n')

i = 16868 # 16899, 16964
nfiles = len(index_ade20k['filename'])
file_name = index_ade20k['filename'][i]
num_obj = index_ade20k['objectPresence'][:, i].sum()
num_parts = index_ade20k['objectIsPart'][:, i].sum()
count_obj = index_ade20k['objectPresence'][:, i].max()
obj_id = np.where(index_ade20k['objectPresence'][:, i] == count_obj)[0][0]
obj_name = index_ade20k['objectnames'][obj_id]
full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i])
print("The dataset has {} images".format(nfiles))
print("The image at index {} is {}".format(i, file_name))
print("It is located at {}".format(full_file_name))
print("It happens in a {}".format(index_ade20k['scene'][i]))
print("It has {} objects, of which {} are parts".format(num_obj, num_parts))
print("The most common object is object {} ({}), which appears {} times".format(obj_name, obj_id, count_obj))



root_path = '..'

# This function reads the image and mask files and generate instance and segmentation
# masks
info = utils.loadAde20K('{}/{}'.format(root_path, full_file_name))
img = cv2.imread(info['img_name'])[:,:,::-1]
seg = cv2.imread(info['segm_name'])[:,:,::-1]
seg_mask = seg.copy()

# The 0 index in seg_mask corresponds to background (not annotated) pixels
seg_mask[info['class_mask'] != obj_id+1] *= 0
plt.figure(figsize=(24,10))

plt.imshow(np.concatenate([img, seg, seg_mask], 1))
plt.axis('off')


print("Objects:",info['objects']['class'])

import os
from glob import glob
from PIL import Image
DATASET_PATH = '../ADE20K_2021_17_01/'
image_label_dir = os.path.join(DATASET_PATH, 'images/ADE', 'validation')
for path in glob(image_label_dir + '/**/*.jpg',recursive=True):
    print(os.path.basename(path))
    break
files = [path for path in glob(image_label_dir + '/**/*.jpg',recursive=True)]

image = cv2.imread(files[1])[:,:,::-1]
label = cv2.imread(files[1].replace('.jpg', '_seg.png'))[:,:,::-1]
#image = image.astype(np.float32)
#label = image.astype(np.int32)

plt.figure(figsize=(24,10))

plt.imshow(np.concatenate([image, label], 1))
plt.axis('off')

import torch
from torchvision import transforms
label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
image = cv2.imread(files[1])[:,:,::-1]
to_tensor = transforms.ToTensor()
image = Image.fromarray(image)
image = to_tensor(image)
print(image.shape,image.dtype)
label_np = label.numpy()
image_np = image.permute(1,2,0).numpy()

f = plt.figure(figsize=(24,10))
plt.subplot(1,2,1)
plt.imshow(image_np)
plt.subplot(1,2,2)
plt.imshow(label_np)
plt.axis('off')  # Hide axes
plt.show()

angle_range = (-10, 10)
MEAN = [0.48897059, 0.46548275, 0.4294]
STD = [0.22861765, 0.22948039, 0.24054667]
# Define the transformation
transform = transforms.Compose([
    transforms.RandomResizedCrop(700),
    transforms.RandomRotation(angle_range),
    transforms.ToTensor(),
    transforms.Normalize(MEAN,STD)
])

i = 0
for i in range(10):
    image = cv2.imread(files[i])[:,:,::-1]
    label = cv2.imread(files[i].replace('.jpg', '_seg.png'))[:,:,::-1]
    image_ = Image.fromarray(image)
    label_ = Image.fromarray(label)
    image_ = transform(image_)
    label_ = transform(label_)
    print(image_.shape,label_.shape)

    f = plt.figure(figsize=(24,10))
    plt.subplot(1,3,1)
    plt.imshow(image_.permute(1,2,0).numpy())
    plt.subplot(1,3,2)
    plt.imshow(label_.permute(1,2,0).numpy())
    plt.subplot(1,3,3)
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()