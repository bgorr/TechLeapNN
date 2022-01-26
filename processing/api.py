import os
import netCDF4
import cv2
import numpy as np
import torch
import pprint
import pickle
from torchvision import transforms
from torch.autograd import Variable




class DataProcessingClient:


    def __init__(self, data_dir=None, save_path=None):

        # --> Training Data <--
        self.data_dir = '/app/data/viirs/training_dataset'  # Default path
        if data_dir is not None:
            self.data_dir = data_dir

        self.save_path = '/app/output/train_dataset_ready_7bands.p'
        if save_path is not None:
            self.save_path = save_path

        self.images_dir = os.path.join(self.data_dir, 'images')
        self.labels_dir = os.path.join(self.data_dir, 'labels')

        self.image_files = [f for f in os.listdir(self.images_dir)]
        self.label_files = [f for f in os.listdir(self.labels_dir)]

        self.pairs = self.get_file_pairs()

        # --> Pytorch Tensors <--
        self.total_image_tensor = torch.Tensor()
        self.total_label_tensor = torch.Tensor()


    def get_file_pairs(self):
        pairs = []
        for image in self.image_files:
            image_id = image[14:22]
            for label in self.label_files:
                label_id = label[25:33]
                if image_id == label_id:
                    pair = (os.path.join(self.images_dir, image), os.path.join(self.labels_dir, label))
                    pairs.append(pair)
                    break
        return pairs

    def print(self, item):
        pprint.PrettyPrinter(indent=4).pprint(item)



    def process_pair(self, pair):
        image_ds = netCDF4.Dataset(pair[0])  # 3248 x 3200 | 3232 x 3200
        label_ds = netCDF4.Dataset(pair[1])  # 406  x 400  | 404  x 400

        print('--> PROCESSING IMAGES / LABELS')
        image_tensor = self.process_image(image_ds)
        label_tensor = self.process_label(label_ds)

        print('--> PATCHING')
        return self.process_patches(image_tensor, label_tensor)


    def process_image(self, image_ds):

        # --> 1. Get image data for all bands: 3D array
        #   - D1: Band
        #   - D2: x-pixel index
        #   - D3: y-pixel index
        bands = image_ds['observation_data'].variables
        img_data = None
        specific_bands = ['M01', 'M03', 'M05', 'M07', 'M09']  # or band == "M11" or band == "M15":  # band == "M03":
        for band in bands:
            # bands --> M01: 0.402-0.422 | M03: 0.478-0.488 | M05: 0.662-0.682 | M07: 0.846-0.885 | M09: 1.371-1.386
            if band in specific_bands:
                band_data = bands[band]  # 3248 x 3200 | 3232 x 3200
                band_data = band_data[:3200, :]

                if img_data is None:
                    img_data = np.array(band_data)
                    img_data = img_data[None]
                else:
                    img_data = np.concatenate((img_data, [np.array(band_data)]))

        # --> 2. Apply mask to 3D array
        img_data = np.ma.masked_greater(img_data, 6.55e4)  # mask outliers
        img_data = np.ma.filled(img_data, np.nan)

        # --> 3. Return array as pytorch tensor
        return torch.as_tensor(img_data)

    def process_label(self, label_ds):

        # --> 1. Get labels in 3D array and reshape to 2D: 406 x 400 x 4 --> 406 x 400
        labels = label_ds['geophysical_data'].variables['Corrected_Optical_Depth_Land']
        labels = np.array(labels)
        labels = labels[:400, :, 1]

        # --> 2. Get quality in 2D array: 406 x 400
        # - (must be 3 for land observations. labels gives aerosol optical depth at X um)
        quality = label_ds['geophysical_data'].variables['Land_Ocean_Quality_Flag']
        quality = np.array(quality)
        quality = quality[:400, :]
        for i in range(np.size(quality, 0)):  # 400
            for j in range(np.size(quality, 1)):  # 400
                if labels[i][j] > 0.3:
                    labels[i][j] = 1.0
                else:
                    labels[i][j] = 0.0

        # --> 3. Reshape labels through interpolation: 400 x 400 --> 3200 x 3200
        labels = cv2.resize(labels, dsize=(3200, 3200), interpolation=cv2.INTER_NEAREST_EXACT)

        # --> 4. Finally get tensor and return
        return torch.as_tensor(labels)

    def process_patches(self, image_tensor, label_tensor):

        # image_tensor: 5 x 3200 x 3200
        # label_tensor: 3200 x 3200

        # --> 1. Get image patches
        print('--> get image patches')
        image_patches = image_tensor.unfold(1, 161, 161).unfold(2, 105, 105)
        image_patches = image_patches.contiguous().view(5, 570, 161, 105)
        image_patches = image_patches.permute(1, 0, 2, 3)

        # --> 2. Get label patches
        print('--> get label patches')
        label_patches = label_tensor.unfold(0, 161, 161).unfold(1, 105, 105)
        label_patches = label_patches.contiguous().view(-1, 1, 161, 105)

        # --> 3. Clean patches of NAN values
        print('--> clean patches', image_patches.size(0))

        # --> Old code
        # i = 0
        # while i < image_patches.size(0):
        #     if image_patches[i, :, :, :].isnan().any() or not torch.isin(1, label_patches[i, :, :, :]).any():
        #     # if image_patches[i, :, :, :].isnan().any():
        #         image_patches = torch.cat([image_patches[0:i, :, :, :], image_patches[i + 1:, :, :, :]])
        #         label_patches = torch.cat([label_patches[0:i, :, :, :], label_patches[i + 1:, :, :, :]])
        #     else:
        #         i = i + 1

        # --> New code
        idx_remove = []
        idx_add = []
        for i in range(image_patches.size(0)):
            if image_patches[i, :, :, :].isnan().any() or not torch.isin(1, label_patches[i, :, :, :]).any():
                idx_remove.append(i)
            else:
                idx_add.append(i)

        image_tensor_slices = []
        label_tensor_slices = []
        for idx in idx_add:
            image_tensor_slices.append(image_patches[idx:idx+1, :, :, :])
            label_tensor_slices.append(label_patches[idx:idx+1, :, :, :])

        image_patches = torch.cat(image_tensor_slices)
        label_patches = torch.cat(label_tensor_slices)




        # --> 4. Normalize image patches
        print('--> normalize patches')
        for i in range(image_patches.size(1)):
            AA = image_patches[:, i, :, :].clone()
            AA = AA.view(image_patches.size(0), -1)
            AA -= AA.mean(1, keepdim=True)[0]
            AA /= AA.std(1, keepdim=True)[0]
            AA = AA.view(image_patches.size(0), 161, 105)
            image_patches[:, i, :, :] = AA

        return image_patches, label_patches





    def build(self):
        all_image_patches = [self.total_image_tensor]
        all_label_patches = [self.total_label_tensor]


        for idx, pair in enumerate(self.pairs):
            print('\n\n', idx, '--------------------------------------')

            image_patches, label_patches = self.process_pair(pair)

            all_image_patches.append(image_patches)
            all_label_patches.append(label_patches)

        self.total_image_tensor = torch.cat(tuple(all_image_patches), 0)
        self.total_label_tensor = torch.cat(tuple(all_label_patches), 0)





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





