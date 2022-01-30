import os
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


def duplicates():


    train_image_files = [f for f in os.listdir('/app/data/viirs/training_dataset/images')]
    print(len(train_image_files), len(list(dict.fromkeys(train_image_files))))

    train_label_files = [f for f in os.listdir('/app/data/viirs/training_dataset/labels')]
    print(len(train_label_files), len(list(dict.fromkeys(train_label_files))))



    test_image_files = [f for f in os.listdir('/app/data/viirs/test_dataset/images')]
    print(len(test_image_files), len(list(dict.fromkeys(test_image_files))))

    test_label_files = [f for f in os.listdir('/app/data/viirs/test_dataset/labels')]
    print(len(test_label_files), len(list(dict.fromkeys(test_label_files))))


    counter = 0
    for test_file in test_image_files:
        if test_file in train_image_files:
            counter += 1
    print('--> IMAGE DUPLICATES:', counter)

    counter = 0
    for test_file in test_label_files:
        if test_file in train_label_files:
            counter += 1
    print('--> LABEL DUPLICATES:', counter)


if __name__ == "__main__":
    duplicates()


