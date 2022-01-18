import os
import netCDF4
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from torchvision.io import read_image
import random
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

import pickle

img_dir = "D:/Documents/VIIRS_downloads/test_dataset/images"
lbl_dir = "D:/Documents/VIIRS_downloads/test_dataset/labels"
# "D:/Documents/VIIRS_downloads/test_dataset"

totalTensor = torch.Tensor()
lbl_totalTensor = torch.Tensor()
for idx in range(len(os.listdir(img_dir))): # len(os.listdir(img_dir))
    img_files = [f for f in os.listdir(img_dir)]
    lbl_files = [g for g in os.listdir(lbl_dir)]
    img_path = os.path.join(img_dir, img_files[idx])
    lbl_path = os.path.join(lbl_dir, lbl_files[idx])
    image_ds = netCDF4.Dataset(img_path)
    bands = image_ds['observation_data'].variables
    img_data = []
    for band in bands:
        if band == "M01" or band == "M03" or band == "M05" or band == "M07" or band == "M09": # or band == "M11" or band == "M15":  # band == "M03":
            band_data = bands[band]
            if img_data == []:
                img_data = np.array(band_data)
                img_data = img_data[None]
            else:
                img_data = np.concatenate((img_data, [np.array(band_data)]))
    label_ds = netCDF4.Dataset(lbl_path)
    labels = label_ds['geophysical_data'].variables['Corrected_Optical_Depth_Land']
    quality = label_ds['geophysical_data'].variables['Land_Ocean_Quality_Flag']
    labels = np.array(labels)
    labels = labels[:, :, 1]
    quality = np.array(quality)
    for i in range(np.size(quality, 0)):
        for j in range(np.size(quality, 1)):
            if (labels[i][j] > 0.3):
                labels[i][j] = 1.0
            else:
                labels[i][j] = 0.0
    img_data = np.ma.masked_greater(img_data, 6.55e4)
    nan_mask = np.ma.getmask(img_data[0, :, :])
    img_data = np.ma.filled(img_data, np.nan)
    new_img = np.zeros(shape=(5,404,400))
    for i in range(5):
        band = img_data[i, :, :]
        band = cv2.resize(band, dsize=(400, 404), interpolation=cv2.INTER_CUBIC)
        new_img[i,:,:] = band
    image = torch.as_tensor(new_img)
    label = torch.as_tensor(labels)
    height = 96
    width = 96
    patches = image.unfold(1, height, height).unfold(2, width, width)
    patches = patches.contiguous().view(5, 16, height, width)
    lbl_patches = label.unfold(0, height, height).unfold(1, width, width)
    lbl_patches = lbl_patches.contiguous().view(-1, 1, height, width)
    patches = patches.permute(1, 0, 2, 3)
    i = 0
    while i < patches.size(0):
        if patches[i, :, :, :].isnan().any():
            patches = torch.cat([patches[0:i, :, :, :], patches[i + 1:-1, :, :, :]])
            lbl_patches = torch.cat([lbl_patches[0:i, :, :, :], lbl_patches[i + 1:-1, :, :, :]])
        else:
            i = i + 1
    # do mean and normalization by band!!! try debugging below code
    for i in range(patches.size(1)):
        AA = patches[:, i, :, :].clone()
        AA = AA.view(patches.size(0), -1)
        AA -= AA.mean(1, keepdim=True)[0]
        AA /= AA.std(1, keepdim=True)[0]
        AA = AA.view(patches.size(0), 161, 105)
        patches[:, i, :, :] = AA

    # blue = patches[idx, 0, :, :].detach().numpy()
    # green = patches[idx, 1, :, :].detach().numpy()
    # red = patches[idx, 2, :, :].detach().numpy()
    # ir = patches[idx, 3, :, :].detach().numpy()
    # swir = patches[idx, 4, :, :].detach().numpy()
    # clouds = patches[idx, 5, :, :].detach().numpy()
    # temp = patches[idx, 6, :, :].detach().numpy()
    # img = np.array([red, blue, green])
    # img = np.moveaxis(img, 0, -1)
    # for i in range(3):
    #     mx = np.max(img[:, :, i])
    #     mn = np.min(img[:, :, i])
    #     img[:, :, i] = (img[:, :, i] - mn) / (mx - mn)
    # cloud_img = patches[idx, 0, :, :]
    # cloud_lbl = lbl_patches[idx, 0, :, :]
    # plt.ion()
    # plt.subplot(2, 3, 1)
    # plt.imshow(img)
    # plt.subplot(2, 3, 2)
    # plt.imshow(cloud_lbl)
    # plt.subplot(2, 3, 3)
    # plt.imshow(ir)
    # plt.subplot(2, 3, 4)
    # plt.imshow(swir)
    # plt.subplot(2, 3, 5)
    # plt.imshow(clouds)
    # plt.subplot(2, 3, 6)
    # plt.imshow(temp)
    # plt.show()
    # plt.close('all')

    totalTensor = torch.cat((totalTensor, patches), 0)
    lbl_totalTensor = torch.cat((lbl_totalTensor, lbl_patches), 0)

dataset = list()
for i in range(totalTensor.size(0)):
    image = totalTensor[i, :, :, :].clone()
    labels = lbl_totalTensor[i, :, :, :].clone()
    target22 = labels.permute(1, 2, 0)
    h, w, k = target22.shape
    target3 = torch.zeros(2, h, w)
    for c in range(2):
        target3[c][target22[:, :, 0] == c] = 1
    dataset.append((image, target3))

filename = './output/test_dataset_coarse.p'
f = open(filename, 'wb')
pickle.dump(dataset, f)
f.close()
print("Done!")
