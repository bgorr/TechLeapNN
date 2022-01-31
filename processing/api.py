import os
import netCDF4
import cv2
import numpy as np
import torch
import pprint
import pickle
from torchvision import transforms
from torch.autograd import Variable
from multiprocessing import Process, Pipe


def _proc():
    return 0



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
        """
        Tensor Image: (B, C, H, W) shape, where:
            - B is a number of images in the batch.
            - C is the channel
            - H is the height
            - W is the width
        """
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






    def build_fast(self):

        # --> 1. Process each file pair in a new process to maximize cpu usage
        jobs = []
        connections = []
        for idx, pair in enumerate(self.pairs):
            if idx > 2:
                break

            print('--> Processing Pair:', idx, '--------------------------------------')
            parent_conn, child_conn = Pipe()
            proc = Process(target=self._build, args=(child_conn, pair))
            proc.start()

            connections.append(parent_conn)
            jobs.append(proc)

        # --> 2. Join all processes
        all_image_patches = [self.total_image_tensor]
        all_label_patches = [self.total_label_tensor]
        for idx, proc in enumerate(jobs):
            print('--> Getting data from proc', idx)
            pair = connections[idx].recv()
            all_image_patches.append(pair[0])
            all_label_patches.append(pair[1])
            proc.join()

        print(len(all_image_patches))
        print(len(all_label_patches))

        # --> 3. Put all in tensor
        self.total_image_tensor = torch.cat(tuple(all_image_patches), 0)
        self.total_label_tensor = torch.cat(tuple(all_label_patches), 0)


    def build(self):

        all_image_patches = [self.total_image_tensor]
        all_label_patches = [self.total_label_tensor]

        for idx, pair in enumerate(self.pairs):
            print('--> Processing Pair:', idx, '--------------------------------------')

            image_patches, label_patches = self.process_pair(pair)

            if image_patches is not None and label_patches is not None:
                all_image_patches.append(image_patches)
                all_label_patches.append(label_patches)
            else:
                print('--> SKIPPING PAIR, NO USABLE PATCHES')

        # --> 3. Put all in tensor
        self.total_image_tensor = torch.cat(tuple(all_image_patches), 0)
        self.total_label_tensor = torch.cat(tuple(all_label_patches), 0)


    def _build(self, conn, pair):

        # --> 1. Process pair and get patches
        image_patches, label_patches = self.process_pair(pair)

        # --> 2. Append patch pair to shared variable
        if image_patches is not None and label_patches is not None:
            result_tuple = (image_patches, label_patches)
            conn.send(result_tuple)
        else:
            conn.send(None)

        return




    def process_pair(self, pair):
        image_ds = netCDF4.Dataset(pair[0])  # 3248 x 3200 | 3232 x 3200
        label_ds = netCDF4.Dataset(pair[1])  # 406  x 400  | 404  x 400

        image_tensor = self.process_image(image_ds)
        label_tensor = self.process_label(label_ds)

        return self.process_patches(image_tensor, label_tensor)

    def process_image(self, image_ds):

        # --> 1. Get image data for all bands: 3D array
        #   - D1: Band
        #   - D2: x-pixel index
        #   - D3: y-pixel index
        bands = image_ds['observation_data'].variables
        img_data = None
        specific_bands = ['M01', 'M03', 'M05', 'M07', 'M09', 'M11', 'M15']  # or band == "M11" or band == "M15":  # band == "M03":
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
        print('--> IMAGE PATCHING')

        # --> 1. Get image patches
        image_patches = image_tensor.unfold(1, 161, 161)
        print(image_patches.size())

        image_patches = image_patches.unfold(2, 105, 105)
        print(image_patches.size())

        # 7 x 570 x 161 x 105
        image_patches = image_patches.contiguous().view(7, 570, 161, 105)
        print(image_patches.size())

        # 570 x 7 x 161 x 105
        image_patches = image_patches.permute(1, 0, 2, 3)
        print(image_patches.size())




        # --> 2. Get label patches
        label_patches = label_tensor.unfold(0, 161, 161).unfold(1, 105, 105)
        label_patches = label_patches.contiguous().view(-1, 1, 161, 105)

        # --> 3. Clean patches of NAN values

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
        image_tensor_slices = []
        label_tensor_slices = []
        for i in range(image_patches.size(0)):
            if not (image_patches[i, :, :, :].isnan().any() or not torch.isin(1, label_patches[i, :, :, :]).any()):
                image_tensor_slices.append(image_patches[i:i + 1, :, :, :])
                label_tensor_slices.append(label_patches[i:i + 1, :, :, :])

        if len(image_tensor_slices) == 0 or len(label_tensor_slices) == 0:
            return None, None

        image_patches = torch.cat(image_tensor_slices)
        label_patches = torch.cat(label_tensor_slices)


        # --> 4. Normalize image patches
        for i in range(image_patches.size(1)):
            AA = image_patches[:, i, :, :].clone()
            AA = AA.view(image_patches.size(0), -1)
            AA -= AA.mean(1, keepdim=True)[0]
            AA /= AA.std(1, keepdim=True)[0]
            AA = AA.view(image_patches.size(0), 161, 105)
            image_patches[:, i, :, :] = AA

        return image_patches, label_patches












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

            """
            Tensor Image: (B, C, H, W) shape, where:
                - B is a number of images in the batch.
                - C is the channel
                - H is the height
                - W is the width
            """

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





