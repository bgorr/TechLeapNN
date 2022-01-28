from processing.api import DataProcessingClient
from neuralnet.api import NeuralNetClient


import numpy as np
import pprint
import cv2
from scipy import interpolate
import torch

def print_imo(item):
    pprint.PrettyPrinter(indent=4).pprint(item)


def mask_test():

    # --> 1. Generate numpy 2D array
    test_arr = [[1.0, 2.2], [3.4, 4.5]]
    np_array = np.array(test_arr)
    print('\n---> NUMPY ARRAY')
    print_imo(np_array)

    # --> 2. Generate mask for values > 3
    np_mask = np.ma.masked_greater(np_array, 3)
    print('\n---> NUMPY MASK')
    print_imo(np_mask)

    # --> 3. Replace masked values with np.nan
    np_array_filled = np.ma.filled(np_mask, np.nan)
    print('\n---> NUMPY MASK FILLED')
    print_imo(np_array_filled)


def interpolate_test():

    # --> 1. Generate numpy 2D array: 3 x 3
    test_arr = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    np_array = np.array(test_arr)
    print('\n---> NUMPY ARRAY')
    print_imo(np_array)

    # --> 2. Interpolate with cv2
    interp_array = cv2.resize(np_array, dsize=(4, 4), interpolation=cv2.INTER_NEAREST_EXACT)
    print('\n---> INTERPOLATED ARRAY')
    print_imo(interp_array)

    # --> 3. Interpolate with scipy
    min = 0
    max = 2
    X = np.linspace(min, max, 3)
    Y = np.linspace(min, max, 3)
    x, y = np.meshgrid(X, Y)
    f = interpolate.interp2d(x, y, np_array, kind='linear')
    Xnew = np.linspace(min, max, 4)
    Ynew = np.linspace(min, max, 4)
    inter_array_np = f(Xnew, Ynew)
    print_imo(inter_array_np)

    return 0


def unfold_test():


    # --> 1. Generate numpy 2D array: 3 x 3
    test_arr = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    np_array = np.array(test_arr)
    print('\n---> NUMPY ARRAY')
    print_imo(np_array)

    # --> 2. Convert to tensor
    tensor = torch.as_tensor(np_array)
    print_imo(tensor)

    # --> 3. Unfold
    patches = tensor.unfold(0, 1, 1)
    print_imo(patches)


    return 0


def reshape_test():

    # --> 1. Generate numpy 2D array: 3 x 3
    test_arr = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    np_array = np.array(test_arr)
    print('\n---> NUMPY ARRAY')
    print_imo(np_array)

    shape_arr = np_array.reshape(2, 2)
    print_imo(shape_arr)



def processing():
    data_dir = '/app/data/viirs/test_dataset'
    save_path = '/app/output/test_dataset_ready_7bands.p'

    processing_client = DataProcessingClient(data_dir=data_dir, save_path=save_path)

    # --> 1. Build
    processing_client.build()

    # --> 2. Run
    processing_client.run()


def neuralnet_test():
    print('--> TESTING NEURAL NET')

    client = NeuralNetClient()

    client.train(epochs=3, save=True, plot=False)







if __name__ == "__main__":
    neuralnet_test()






















