import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import thermal_depth_preprocessing, transform, convert_16bit_to_8bit
import torchvision.transforms.functional as FT


class ThermalDataset(Dataset):
    """
    A pytorch Dataset class to be used in a Pytorch Dataloader to create bacthes.
    """
    def __init__(self, data_folder, img_type, split, mean_std=None, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored following this path:
        .
        |-- Arrays (of the fusion of the depth-thermal image)
            |-- fusion1.npy
             -- fusion2.npy
        |-- Annotations
            |-- fusion1.xml
             -- fusion2.xml
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect.
        """
        self.img_type = img_type
        assert self.img_type in {'thermal', 'depth'}

        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_' + self.img_type + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_' + self.img_type + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        assert len(self.images) == len(self.objects)

        if mean_std:
            self.dataset_mean = torch.as_tensor(mean_std[0]).type('torch.FloatTensor')
            self.dataset_std = torch.as_tensor(mean_std[1]).type('torch.FloatTensor')
        else:
            self.dataset_mean, self.dataset_std = self.dataset_mean_std()

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        #print(self.images[i])
        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1-difficulties]
            labels = labels[1-difficulties]
            difficulties = difficulties[1-difficulties]

        # Apply transformation
        image, boxes = thermal_depth_preprocessing(image,
                                                   self.dataset_mean,
                                                   self.dataset_std,
                                                   split=self.split,
                                                   bbox=boxes)
        return image, boxes, labels, difficulties
        #image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
        #return image.type('torch.FloatTensor'), boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties   # tensor (N, 1, 300, 300), 3 lists of N tensors each

    def dataset_mean_std(self):

        mean = 0
        std = 0
        tot_img = 0

        for img in self.images:
            img = Image.open(img, mode='r')
            img = FT.to_tensor(img).type(torch.FloatTensor)
            mean += img.mean()
            std += img.std()
            tot_img += 1

        mean /= tot_img
        std /= tot_img
        return mean, std


class DetectDataset(Dataset):
    """
    A pytorch Dataset class to be used in a Pytorch Dataloader to load test examples.
    """
    def __init__(self, data_folder, mean, std):
        """
        :param data_folder: folder where data files are stored following this path:
        .
        |-- Thermique (of the fusion of the depth-thermal image)
            |-- thermal1.png
             -- thermal2.png
        |-- Thermique_8bit
            |-- thermal1.png
             -- fusion2.png
        :param mean: mean of the original train dataset (used for image standardization)
        :param std: standard deviation of the original train dataset (used for images standardization)
        """
        self.data_folder = data_folder
        self.mean = torch.as_tensor(mean).type('torch.FloatTensor')
        self.std = torch.as_tensor(std).type('torch.FloatTensor')
        self.images = [os.path.join(data_folder, img) for img in os.listdir(data_folder)]

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        original_width = image.width
        original_height = image.height

        image_8bit = convert_16bit_to_8bit(self.images[i])

        # Apply transformation to the 16 bit image
        image = thermal_depth_preprocessing(image,
                                            split='detect',
                                            mean=self.mean,
                                            std=self.std)

        return image.type('torch.FloatTensor'), image_8bit, original_width, original_height

    def __len__(self):
        return len(self.images)






