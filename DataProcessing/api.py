import os
import netCDF4
import cv2
import numpy as np
import torch
from torchvision import transforms





class DataProcessingClient:

    def __init__(self, images_dir=None, labels_dir=None, save_path=None):

        # --> Training Data <--
        self.images_dir = 'D:/Documents/VIIRS_downloads/test_dataset/images'
        self.labels_dir = 'D:/Documents/VIIRS_downloads/test_dataset/labels'
        self.save_path = './output/train_dataset_ready_7bands.p'
        if images_dir is not None:
            self.images_dir = images_dir
        if labels_dir is not None:
            self.labels_dir = labels_dir
        if save_path is not None:
            self.save_path = save_path

        self.image_files = [f for f in os.listdir(self.images_dir)]
        self.label_files = [f for f in os.listdir(self.labels_dir)]

        self.images = [os.path.join(self.images_dir, f) for f in self.image_files]
        self.labels = [os.path.join(self.labels_dir, f) for f in self.label_files]


        # --> Pytorch Tensors <--
        self.total_image_tensor = torch.Tensor()
        self.total_label_tensor = torch.Tensor()


    def _process_image(self, image_file):
        image_ds = netCDF4.Dataset(image_file)

        # --> 1. Get band data
        bands = image_ds['observation_data'].variables
        img_data = []
        specific_bands = ['M01', 'M03', 'M05', 'M07', 'M09']  # or band == "M11" or band == "M15":  # band == "M03":
        for band in bands:
            # bands:
            # M01: 0.402-0.422
            # M03: 0.478-0.488
            # M05: 0.662-0.682
            # M07: 0.846-0.885
            # M09: 1.371-1.386
            if band in specific_bands:
                band_data = bands[band]
                if img_data == []:
                    img_data = np.array(band_data)
                    img_data = img_data[None]
                else:
                    img_data = np.concatenate((img_data, [np.array(band_data)]))

        # --> 2. Apply mask to image
        img_data = np.ma.masked_greater(img_data, 6.55e4)  # mask outliers
        nan_mask = np.ma.getmask(img_data[0, :, :])
        img_data = np.ma.filled(img_data, np.nan)

        # --> 3. Get image tensor
        image = torch.as_tensor(img_data)

        # --> 4. Crop image from 3232x3200 to 3200x3200
        p = transforms.Compose([transforms.CenterCrop([3200, 3200])])
        image = p(image)

        # --> 5. Return image
        return image

    def _process_label(self, label_file):
        label_ds = netCDF4.Dataset(label_file)

        # --> 1. Get labels
        labels = label_ds['geophysical_data'].variables['Corrected_Optical_Depth_Land']
        labels = np.array(labels)
        labels = labels[:, :, 1]

        # --> 2. Get quality (must be 3 for land observations. labels gives aerosol optical depth at X um)
        quality = label_ds['geophysical_data'].variables['Land_Ocean_Quality_Flag']
        quality = np.array(quality)
        for i in range(np.size(quality, 0)):
            for j in range(np.size(quality, 1)):
                if quality == 3.0 and labels[i][j] > 0.3:
                    labels[i][j] = 1.0
                else:
                    labels[i][j] = 0.0

        # --> 3. Resize labels from 404x400 to 3200x3200
        labels = cv2.resize(labels, dsize=(3200, 3200), interpolation=cv2.INTER_NEAREST_EXACT)

        # --> 4. Finally get tensor and return
        return torch.as_tensor(labels)


    def _image_patches(self, image):
        patches = image.unfold(1, 161, 161).unfold(2, 105, 105)
        patches = patches.contiguous().view(5, 570, 161, 105)
        return patches.permute(1, 0, 2, 3)

    def _label_patches(self, label):
        lbl_patches = label.unfold(0, 161, 161).unfold(1, 105, 105)
        return lbl_patches.contiguous().view(-1, 1, 161, 105)


    def clean_patches(self, image_patches, label_patches):
        i = 0
        while i < image_patches.size(0):
            if image_patches[i, :, :, :].isnan().any() or not torch.isin(1, label_patches[i, :, :, :]).any():
                image_patches = torch.cat([image_patches[0:i, :, :, :], image_patches[i + 1:-1, :, :, :]])
                label_patches = torch.cat([label_patches[0:i, :, :, :], label_patches[i + 1:-1, :, :, :]])
            else:
                i = i + 1

    def normalize_patches(self, image_patches, label_patches):
        for i in range(image_patches.size(1)):
            AA = image_patches[:, i, :, :].clone()
            AA = AA.view(patches.size(0), -1)
            AA -= AA.mean(1, keepdim=True)[0]
            AA /= AA.std(1, keepdim=True)[0]
            AA = AA.view(patches.size(0), 161, 105)
            image_patches[:, i, :, :] = AA

    def build(self):

        # --> Iterate over image / label pairs
        for idx in range(len(self.images)):

            # --> 1. Extract data from image / label files
            image = self._process_image(self.images[idx])
            label = self._process_label(self.labels[idx])

            # --> 2. Get image / label patches
            image_patches = self._image_patches(image)
            label_patches = self._label_patches(label)

            # --> 3. Remove nan values from patches
            self.clean_patches(image_patches, label_patches)

            # --> 4. Normalize patch values
            self.normalize_patches(image_patches, label_patches)

            # --> 5. Index patch tensor to total tensors
            self.total_image_tensor = torch.cat((self.total_image_tensor, image_patches), 0)
            self.total_label_tensor = torch.cat((self.total_label_tensor, label_patches), 0)

    def run(self):

        # --> 1. Reshape tensors for neural net, one hot encoding
        W = torch.empty(1, 1, 3, 3)
        W[0, 0, :, :] = torch.Tensor([[.111, .111, .111], [.111, .111, .111], [.111, .111, .111]])  # .111~1/9
        W = torch.nn.Parameter(W)
        my_conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=False)
        my_conv.weight = W

        # --> 2. Pickle dataset and save to file
        dataset = []
        for i in range(self.total_image_tensor.size(0)):
            image_data = self.total_image_tensor[i, :, :, :].clone()
            label_data = self.total_label_tensor[i, :, :, :].clone()

            image_data = Variable(image_data).float()
            label_data = Variable(label_data).float()

            # --> 3. Get target data from label data
            target0 = torch.FloatTensor(label_data)
            target1 = my_conv(torch.unsqueeze(target0.permute(0, 1, 2), 0))
            target2 = (target1 > 0.5).float() * 1
            target22 = target2[0, :, :, :].permute(1, 2, 0)
            h, w, k = target22.shape
            target3 = torch.zeros(2, h, w)
            for c in range(2):
                target3[c][target22[:, :, 0] == c] = 1

            # --> 4. Append to final dataset
            dataset.append((image_data, target3))


        # --> 5. Pickle final dataset and save
        f = open(self.save_path, 'wb')
        pickle.dump(dataset, f)
        f.close()

