import os
import netCDF4
import cv2
import numpy as np
import torch
import glob
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')

import pickle
import pandas as pd

# you'll have to point these to your dataset location
img_dir = "/home/ben/Documents/landsat_downloads/scene1/landsat_bands"
lbl_path = "/home/ben/Documents/landsat_downloads/scene1/cirrus_groundtruth.tif"

totalTensor = torch.Tensor()
lbl_totalTensor = torch.Tensor()
bands = np.zeros(shape=(7911, 7781, 11))
for string in os.listdir(img_dir):  # len(os.listdir(img_dir))
    print(string)
    substr = string[42:44]
    if "." in substr:
        substr = substr[0:1]
    band = int(substr)
    if band == 8:
        continue
    band_path = os.path.join(img_dir, string)
    band_data = Image.open(band_path)
    band_array = np.array(band_data)
    bands[:, :, band-1] = band_array
bands = np.delete(bands, [0,5,6,7,8,9,10], 2)
#bands = np.delete(bands, 4, 2)
#bands = np.delete(bands, 7, 2)
label = Image.open(lbl_path)
label_array = np.array(label)

image = torch.as_tensor(bands)
label = torch.as_tensor(label_array)

# convert 3200x3200 images into 570 161x105 images
# label = label.permute(1, 0)
height = 512
width = 512
image = image.permute(2,0,1)
patches = image.unfold(1, height, height).unfold(2, width, width)
patches = patches.contiguous().view(4, patches.size(1)*patches.size(2), height, width)
lbl_patches = label.unfold(0, height, height).unfold(1, width, width)
lbl_patches = lbl_patches.contiguous().view(-1, 1, height, width)
patches = patches.permute(1, 0, 2, 3)
#truth = torch.isin(1,label).any().item()
truth = torch.isin(1.0,lbl_patches).any().item()
# eliminate entries with nan values (can't pass nan values into neural net)
image_tensor_slices = []
label_tensor_slices = []
for i in range(patches.size(0)):
    if not patches[i, :, :, :].isnan().any():
        if torch.isin(1, lbl_patches[i, :, :, :]).any():
            image_tensor_slices.append(patches[i:i + 1, :, :, :])
            label_tensor_slices.append(lbl_patches[i:i + 1, :, :, :])
        else:
            print('--> IMAGE PATCH CONTAINS NO TRUE VALUES')
    else:
        print('--> IMAGE PATCH CONTAINS NANs')

patches = torch.cat(image_tensor_slices)
lbl_patches = torch.cat(label_tensor_slices)
# do mean and unit variance by band
for i in range(patches.size(1)):
    AA = patches[:, i, :, :].clone()
    AA = AA.view(patches.size(0), -1)
    AA -= AA.mean(1, keepdim=True)[0]
    std = AA.std(1, keepdim=True)[0]
    if std != 0.0:
        AA /= AA.std(1, keepdim=True)[0]
    AA = AA.view(patches.size(0), height, width)
    patches[:, i, :, :] = AA

# plotting for sanity check
blue = patches[0, 0, :, :].detach().numpy()
green = patches[0, 1, :, :].detach().numpy()
red = patches[0, 2, :, :].detach().numpy()
ir = patches[0, 3, :, :].detach().numpy()
# clouds = patches[idx, 5, :, :].detach().numpy()
# temp = patches[idx, 6, :, :].detach().numpy()
img = np.array([red, blue, green])
img = np.moveaxis(img, 0, -1)
for i in range(3):
    mx = np.max(img[:, :, i])
    mn = np.min(img[:, :, i])
    img[:, :, i] = (img[:, :, i] - mn) / (mx - mn)
cloud_img = patches[0, 0, :, :]
cloud_lbl = lbl_patches[0, 0, :, :]
plt.ion()
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(cloud_lbl)
plt.subplot(2, 2, 3)
plt.imshow(ir)
plt.subplot(2, 2, 4)
# plt.subplot(2, 3, 5)
# plt.imshow(clouds)
# plt.subplot(2, 3, 6)
# plt.imshow(temp)
plt.show()
plt.close('all')

totalTensor = torch.cat((totalTensor, patches), 0)
lbl_totalTensor = torch.cat((lbl_totalTensor, lbl_patches), 0)

# reshape tensors for neural net, one hot encoding
W = torch.empty(1, 1, 3, 3)
W[0, 0, :, :] = torch.Tensor([[.111, .111, .111], [.111, .111, .111], [.111, .111, .111]])  # .111~1/9
W = torch.nn.Parameter(W)
my_conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=False)
my_conv.weight = W
# dataset = list()
# for i in range(totalTensor.size(0)):
#     image = totalTensor[i, :, :, :].clone()
#     labels = lbl_totalTensor[i, :, :, :].clone()
#     target22 = labels.permute(1, 2, 0)
#     h, w, k = target22.shape
#     target3 = torch.zeros(2, h, w)
#     for c in range(2):
#         target3[c][target22[:, :, 0] == c] = 1
#     dataset.append((image, target3))

dataset = list()
for i in range(totalTensor.size(0)):
    dat = totalTensor[i, :, :, :].clone()
    target = lbl_totalTensor[i, :, :, :].clone()
    dat, target = Variable(dat).float(), Variable(target).float()
    target0 = torch.FloatTensor(target)
    target1 = my_conv(torch.unsqueeze(target0.permute(0, 1, 2), 0))
    target2 = (target1 > 0.5).float() * 1
    target22 = target2[0, :, :, :].permute(1, 2, 0)
    h, w, k = target22.shape
    target3 = torch.zeros(2, h, w)
    for c in range(2):
        target3[c][target22[:, :, 0] == c] = 1
    dataset.append((dat, target3))

# save dataset
for i in range(len(dataset)):
    blue = pd.DataFrame(dataset[i][0][0].numpy())
    green = pd.DataFrame(dataset[i][0][1].numpy())
    red = pd.DataFrame(dataset[i][0][2].numpy())
    nir = pd.DataFrame(dataset[i][0][3].numpy())
    blue.to_csv(str(i)+'_blue.csv', index=False, header=False)
    green.to_csv(str(i) + '_green.csv', index=False, header=False)
    red.to_csv(str(i) + '_red.csv', index=False, header=False)
    nir.to_csv(str(i) + '_nir.csv', index=False, header=False)
filename = './output/landsat_dataset_scene1_cirrus_5bands.p'
f = open(filename, 'wb')
pickle.dump(dataset, f)
f.close()
print("Done!")
