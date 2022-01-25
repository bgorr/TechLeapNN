from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

tiff_file = '/home/ben/Documents/landsat_downloads/scene1/landsat_bands/LC08_L1TP_034032_20200819_20200904_02_T1_B2.TIF'
cirrus_tiff_file = '/home/ben/Documents/landsat_downloads/scene1/xd_cirrus.tif'
cloud_tiff_file = '/home/ben/Documents/landsat_downloads/scene1/xd_cloud.tif'
im = Image.open(tiff_file)
cirrus_im = Image.open(cirrus_tiff_file)
cloud_im = Image.open(cloud_tiff_file)
imarray = np.array(im)
cirrus_array = np.array(cirrus_im)
cloud_array = np.array(cloud_im)
groundtruth = np.zeros(shape=(7911, 7781))
for i in range(5250, 6250):
    for j in range(500, 1750):
        if cirrus_array[i, j] == 1 and cloud_array[i, j] == 0:
            groundtruth[i, j] = 1
groundtruthim = Image.fromarray(groundtruth)
groundtruthim.save('/home/ben/Documents/landsat_downloads/scene1/groundtruth.tif')
plt.imshow(imarray)
plt.show()
plt.close()
plt.imshow(cirrus_array)
plt.show()
plt.close()
plt.imshow(cloud_array)
plt.show()
plt.close()
plt.imshow(groundtruth)
plt.show()
plt.close()
