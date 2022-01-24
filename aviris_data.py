import numpy as np
import sys
#import tensorflow as tf
#from tensorflow import keras
import cv2
#import struct
from spectral import *
#import spectral.io.envi as envi

np.set_printoptions(threshold = sys.maxsize)


#### Useful functions
def find_clip_val_channel(channel_ndarray):
    n_rows = len(channel_ndarray)
    clip_val = sorted(set((channel_ndarray[0, :])))[1] - 1e-3
    for i in range(1, n_rows):
        clip_val_row = sorted(set((channel_ndarray[i, :])))[1] - 1e-3
        if clip_val_row < clip_val:
            clip_val = clip_val_row

    return clip_val

#### Unpack image file (binary 32-bit little-endian floating point IEEE)
filepath = 'D:\\Downloads\\AVIRIS data\\' # change to actual hard drive path
img_names = ['ang20170618t194516_corr_v2p7_img']
header_names = [s + '.hdr' for s in img_names]
header_filepath = filepath + header_names[0]

#view_indexed(img)
#view = imshow(img, bands=(30, 20, 10))
#view_cube(img, bands=[29, 19, 9])

f = open(header_filepath, "r")
# file_data = f.read()
line_to_read = 14

for position, line in enumerate(f):
    if position == line_to_read:
        print(line)

# print(file_data[19])

#### Read and store processed images
corrected_images_dict = {}

for i in range(len(img_names)):
    img_filepath = filepath + img_names[i]
    header_filepath = filepath + header_names[i]

    img = io.envi.open(header_filepath, img_filepath)
    # print(img)

    rgb_corr_vals = np.array(img.read_bands((51, 31, 17)))
    print(np.array(rgb_corr_vals).shape)

    rgb_img_vals = np.zeros((len(rgb_corr_vals), len(rgb_corr_vals[0]), 3))

    max_corr_red = np.max(rgb_corr_vals[:, :, 0])
    max_corr_green = np.max(rgb_corr_vals[:, :, 1])
    max_corr_blue = np.max(rgb_corr_vals[:, :, 2])
    max_corr = [max_corr_red, max_corr_green, max_corr_blue]
    clip_val_red = find_clip_val_channel(rgb_corr_vals[:, :, 0])
    clip_val_green = find_clip_val_channel(rgb_corr_vals[:, :, 1])
    clip_val_blue = find_clip_val_channel(rgb_corr_vals[:, :, 2])
    min_corr = [clip_val_red, clip_val_green, clip_val_blue]

    for j in range(3):
        corr_color_vals = rgb_corr_vals[:, :, j]
        corr_color_vals = np.clip(corr_color_vals, a_min=min_corr[j], a_max=max_corr[j])
        rgb_img_vals[:, :, j] = np.multiply(
            np.divide(np.subtract(corr_color_vals, min_corr[j]), max_corr[j] - min_corr[j]), 255)

    dict_key = 'img' + str(i)
    corrected_images_dict[dict_key] = rgb_img_vals.astype(int)

    #view = imshow(rgb_img_vals.astype(int))
    print(view)

#### Preprocesss images
preprocessed_images = {}
for i in range(len(img_names)):
    dict_key = 'img' + str(i)
    current_image = corrected_images_dict[dict_key]

    current_image_preprocessed = np.zeros((np.size(current_image,0), np.size(current_image,1), 3))
    for j in range(3):
        current_image_preprocessed[:, :, j] = np.divide(
            np.subtract(current_image[:, :, j], np.mean(current_image[:, :, j])), np.std(current_image[:, :, j]))

    preprocessed_images[dict_key] = current_image_preprocessed
    #imshow(current_image_preprocessed)
numRows = 20
numCols = 4
sizeX = np.size(rgb_corr_vals, 1)
sizeY = np.size(rgb_corr_vals, 0)
print(np.floor(sizeY/numRows))
print(np.floor(sizeX/numCols))
image_dataset = np.zeros(shape=(numRows*numCols,int(np.floor(sizeY/numRows)),int(np.floor(sizeX/numCols)), 3))
idx = 0
#img = cv2.imread(rgb_img_vals)
for i in range(numRows):
    for j in range(numCols):
        roi = rgb_img_vals[int(i*sizeY/numRows):int(i*sizeY/numRows + sizeY/numRows), int(j*sizeX/numCols):int(j*sizeX/numCols + sizeX/numCols), :]
        image_dataset[idx,:,:,:] = roi
        idx = idx + 1
len(rgb_corr_vals)
len(rgb_corr_vals[0])
len(rgb_corr_vals[0][0])

s = set(rgb_corr_vals[0,:,2])
print(sorted(s))