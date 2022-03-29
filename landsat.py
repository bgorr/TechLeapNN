from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

blue_tiff_file = '/home/ben/Documents/landsat_downloads/scene1/landsat_bands/LC08_L1TP_034032_20200819_20200904_02_T1_B2.TIF'
green_tiff_file = '/home/ben/Documents/landsat_downloads/scene1/landsat_bands/LC08_L1TP_034032_20200819_20200904_02_T1_B3.TIF'
red_tiff_file = '/home/ben/Documents/landsat_downloads/scene1/landsat_bands/LC08_L1TP_034032_20200819_20200904_02_T1_B4.TIF'
# cirrus_tiff_file = '/home/ben/Documents/landsat_downloads/scene2/xd_cirrus.tif'
# cloud_tiff_file = '/home/ben/Documents/landsat_downloads/scene2/xd_cloud.tif'
label_tiff_file = '/home/ben/Documents/landsat_downloads/scene1/labels_json/labels.tif'
blue_im = Image.open(blue_tiff_file)
green_im = Image.open(green_tiff_file)
red_im = Image.open(red_tiff_file)
# cirrus_im = Image.open(cirrus_tiff_file)
# cloud_im = Image.open(cloud_tiff_file)
label_im = Image.open(label_tiff_file)
blue_array = np.array(blue_im)
green_array = np.array(green_im)
red_array = np.array(red_im)
# cirrus_array = np.array(cirrus_im)
# cloud_array = np.array(cloud_im)
label_array = np.array(label_im)
img = np.array([red_array, blue_array, green_array], float)
img = np.moveaxis(img, 0, -1)
for n in range(3):
    mx = np.max(img[:, :, n])
    mn = np.min(img[:, :, n])
    img[:, :, n] = (img[:, :, n] - mn) / (mx - mn)
groundtruth = np.zeros(shape=(7911, 7781))
for i in range(7861):
    for j in range(7731):
        if label_array[i, j] == 255:
            groundtruth[i, j] = 1
        # if cirrus_array[i, j] == 1 and cloud_array[i, j] == 0:
        #     groundtruth[i, j] = 1
groundtruthim = Image.fromarray(groundtruth)
groundtruthim.save('/home/ben/Documents/landsat_downloads/scene1/manual_groundtruth.tif')
plt.imshow(img)
plt.show()
plt.close()
# plt.imshow(cirrus_array)
# plt.show()
# plt.close()
# plt.imshow(cloud_array)
# plt.show()
# plt.close()
plt.imshow(groundtruth)
plt.show()
plt.close()
