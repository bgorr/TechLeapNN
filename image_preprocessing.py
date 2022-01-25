import os
import netCDF4
import cv2
import numpy as np
import torch
import glob
from torchvision import transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')

import pickle

# you'll have to point these to your dataset location
img_dir = "/home/ben/Documents/VIIRS_downloads/training_dataset/images"
lbl_dir = "/home/ben/Documents/VIIRS_downloads/training_dataset/labels"

totalTensor = torch.Tensor()
lbl_totalTensor = torch.Tensor()
for idx in range(len(os.listdir(img_dir))):  # len(os.listdir(img_dir))
    print(idx)
    img_files = [f for f in os.listdir(img_dir)]
    lbl_files = [g for g in os.listdir(lbl_dir)]
    substring = [img_files[idx][14:22]]
    final_list = [nm for ps in substring for nm in lbl_files if ps in nm]
    lbl_path = final_list[0]
    if not lbl_path:
        continue
    img_path = os.path.join(img_dir, img_files[idx])
    lbl_path = os.path.join(lbl_dir, lbl_path)
    image_ds = netCDF4.Dataset(img_path)
    bands = image_ds['observation_data'].variables
    img_data = []
    for band in bands:
        # bands:
        # M01: 0.402-0.422
        # M03: 0.478-0.488
        # M05: 0.662-0.682
        # M07: 0.846-0.885
        # M09: 1.371-1.386
        if band == "M01" or band == "M03" or band == "M05" or band == "M07" or band == "M09" or band == "M11" or band == "M15":  # band == "M03":
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
    # quality must be 3 for land observations. labels gives aerosol optical depth at X um
    for i in range(np.size(quality, 0)):
        for j in range(np.size(quality, 1)):
            if labels[i][j] > 0.3:
                labels[i][j] = 1.0
            else:
                labels[i][j] = 0.0

    # resize labels from 404x400 to 3200x3200
    labels = cv2.resize(labels, dsize=(3200, np.size(img_data, 1)), interpolation=cv2.INTER_NEAREST_EXACT)
    img_data = np.ma.masked_greater(img_data, 6.55e4)  # mask outliers
    nan_mask = np.ma.getmask(img_data[0, :, :])
    img_data = np.ma.filled(img_data, np.nan)
    ind = 0
    while ind < np.size(img_data, 1):
        if np.isnan(img_data[0, ind, 0]):
            img_data = np.delete(img_data, ind, 1)
            labels = np.delete(labels, ind, 0)
        else:
            ind = ind + 1
    image = torch.as_tensor(img_data)
    label = torch.as_tensor(labels)
    # p = transforms.Compose([transforms.CenterCrop([3200, 3200])]) # crop from 3232x3200 to 3200x3200
    # image = p(image)
    # label = p(label)

    # # plot image
    # full_img = np.array(image[0,:, :])
    # plt.imshow(full_img)
    # plt.show()
    # plt.close()

    # convert 3200x3200 images into 570 161x105 images
    # label = label.permute(1, 0)
    patches = image.unfold(1, 161, 161).unfold(2, 105, 105)
    patches = patches.contiguous().view(7, 450, 161, 105)
    lbl_patches = label.unfold(0, 161, 161).unfold(1, 105, 105)
    lbl_patches = lbl_patches.contiguous().view(-1, 1, 161, 105)
    patches = patches.permute(1, 0, 2, 3)

    # eliminate entries with nan values (can't pass nan values into neural net)
    i = 0
    while i < patches.size(0):
        if patches[i, :, :, :].isnan().any() or not torch.isin(1, lbl_patches[i, :, :, :]).any():
            patches = torch.cat([patches[0:i, :, :, :], patches[i + 1:-1, :, :, :]])
            lbl_patches = torch.cat([lbl_patches[0:i, :, :, :], lbl_patches[i + 1:-1, :, :, :]])
        else:
            i = i + 1
    if patches.size(0) == 0:
        continue
    # do mean and unit variance by band
    for i in range(patches.size(1)):
        AA = patches[:, i, :, :].clone()
        AA = AA.view(patches.size(0), -1)
        AA -= AA.mean(1, keepdim=True)[0]
        AA /= AA.std(1, keepdim=True)[0]
        AA = AA.view(patches.size(0), 161, 105)
        patches[:, i, :, :] = AA

    # plotting for sanity check
    blue = patches[0, 0, :, :].detach().numpy()
    green = patches[0, 1, :, :].detach().numpy()
    red = patches[0, 2, :, :].detach().numpy()
    ir = patches[0, 3, :, :].detach().numpy()
    swir = patches[0, 4, :, :].detach().numpy()
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
    plt.imshow(swir)
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
filename = './output/train_dataset_ready_7bands.p'
f = open(filename, 'wb')
pickle.dump(dataset, f)
f.close()
print("Done!")
