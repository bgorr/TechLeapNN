import os
import netCDF4
import cv2
import numpy as np
import torch
from torchvision import transforms

import pickle

# you'll have to point these to your dataset location
img_dir = "D:/Documents/VIIRS_downloads/test_dataset/images"
lbl_dir = "D:/Documents/VIIRS_downloads/test_dataset/labels"

totalTensor = torch.Tensor()
lbl_totalTensor = torch.Tensor()
for idx in range(len(os.listdir(img_dir))):
    img_files = [f for f in os.listdir(img_dir)]
    lbl_files = [g for g in os.listdir(lbl_dir)]
    img_path = os.path.join(img_dir, img_files[idx])
    lbl_path = os.path.join(lbl_dir, lbl_files[idx])
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
    # quality must be 3 for land observations. labels gives aerosol optical depth at X um
    for i in range(np.size(quality, 0)):
        for j in range(np.size(quality, 1)):
            if quality == 3.0 and labels[i][j] > 0.3:
                labels[i][j] = 1.0
            else:
                labels[i][j] = 0.0

    # resize labels from 404x400 to 3200x3200
    labels = cv2.resize(labels, dsize=(3200, 3200), interpolation=cv2.INTER_NEAREST_EXACT)
    img_data = np.ma.masked_greater(img_data, 6.55e4) # mask outliers
    nan_mask = np.ma.getmask(img_data[0, :, :])
    img_data = np.ma.filled(img_data, np.nan)
    image = torch.as_tensor(img_data)
    label = torch.as_tensor(labels)
    p = transforms.Compose([transforms.CenterCrop([3200, 3200])]) # crop from 3232x3200 to 3200x3200
    image = p(image)

    # # plot image
    # full_img = np.array(image[0,:, :])
    # plt.imshow(full_img)
    # plt.show()
    # plt.close()

    # convert 3200x3200 images into 570 161x105 images
    patches = image.unfold(1, 161, 161).unfold(2, 105, 105)
    patches = patches.contiguous().view(5, 570, 161, 105)
    lbl_patches = label.unfold(0, 161, 161).unfold(1, 105, 105)
    lbl_patches = lbl_patches.contiguous().view(-1, 1, 161, 105)
    patches = patches.permute(1, 0, 2, 3)

    # eliminate entries with nan values (can't pass nan values into neural net)
    i = 0
    while i < patches.size(0):
        if patches[i, :, :, :].isnan().any():
            patches = torch.cat([patches[0:i, :, :, :], patches[i + 1:-1, :, :, :]])
            lbl_patches = torch.cat([lbl_patches[0:i, :, :, :], lbl_patches[i + 1:-1, :, :, :]])
        else:
            i = i + 1

    # do mean and unit variance by band
    for i in range(patches.size(1)):
        AA = patches[:, i, :, :].clone()
        AA = AA.view(patches.size(0), -1)
        AA -= AA.mean(1, keepdim=True)[0]
        AA /= AA.std(1, keepdim=True)[0]
        AA = AA.view(patches.size(0), 161, 105)
        patches[:, i, :, :] = AA

    # # plotting for sanity check
    # blue = patches[idx, 0, :, :].detach().numpy()
    # green = patches[idx, 1, :, :].detach().numpy()
    # red = patches[idx, 2, :, :].detach().numpy()
    # ir = patches[idx, 3, :, :].detach().numpy()
    # swir = patches[idx, 4, :, :].detach().numpy()
    # # clouds = patches[idx, 5, :, :].detach().numpy()
    # # temp = patches[idx, 6, :, :].detach().numpy()
    # img = np.array([red, blue, green])
    # img = np.moveaxis(img, 0, -1)
    # for i in range(3):
    #     mx = np.max(img[:, :, i])
    #     mn = np.min(img[:, :, i])
    #     img[:, :, i] = (img[:, :, i] - mn) / (mx - mn)
    # cloud_img = patches[idx, 0, :, :]
    # cloud_lbl = lbl_patches[idx, 0, :, :]
    # plt.ion()
    # plt.subplot(2, 2, 1)
    # plt.imshow(img)
    # plt.subplot(2, 2, 2)
    # plt.imshow(cloud_lbl)
    # plt.subplot(2, 2, 3)
    # plt.imshow(ir)
    # plt.subplot(2, 2, 4)
    # plt.imshow(swir)
    # # plt.subplot(2, 3, 5)
    # # plt.imshow(clouds)
    # # plt.subplot(2, 3, 6)
    # # plt.imshow(temp)
    # plt.show()
    # plt.close('all')

    totalTensor = torch.cat((totalTensor, patches), 0)
    lbl_totalTensor = torch.cat((lbl_totalTensor, lbl_patches), 0)

# reshape tensors for neural net, one hot encoding
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

# save dataset
filename = './output/test_dataset_ready.p'
f = open(filename, 'wb')
pickle.dump(dataset, f)
f.close()
print("Done!")
