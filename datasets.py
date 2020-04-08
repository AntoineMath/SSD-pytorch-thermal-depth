import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform, convert_16bit_to_8bit
import torchvision.transforms.functional as FT


class ThermalDepthDataset(Dataset):
    """
    A pytorch Dataset class to be used in a Pytorch Dataloader to create bacthes.
    """
    def __init__(self, data_folder, split, mean_std=None, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored following this path:
        .
        |-- Thermal (of the fusion of the depth-thermal image)
            |-- thermal1.png
             -- thermal2.png
        |-- Depth
            |-- depth1.png
             -- depth2.png
        |-- Annotations
            |-- thermal1.xml
             -- thermal2.xml
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect.
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.thermal_images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        assert len(self.thermal_images) == len(self.objects)

        self.depth_images = [img.replace('Thermique', 'Profondeur').replace('thermal', 'depth') for img in self.thermal_images]
        assert len(self.thermal_images) == len(self.depth_images), 'some thermal images names are not matching the depth images ones'

        # compute the mean and std of the dataset
        if mean_std:
            self.dataset_mean, self.dataset_std = mean_std
        else:
            self.dataset_mean, self.dataset_std = self.dataset_mean_std()

    def __getitem__(self, i):
        # Read image
        thermal_image = Image.open(self.thermal_images[i], mode='r')
        depth_image = Image.open(self.depth_images[i], mode='r')

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
        thermal_image, depth_image, boxes, labels, difficulties = transform(thermal_image, depth_image,
                                                                            split=self.split,
                                                                            boxes=boxes,
                                                                            labels=labels,
                                                                            difficulties=difficulties)

        return thermal_image, depth_image, boxes, labels, difficulties

    def __len__(self):
        return len(self.thermal_images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        thermal_images = list()
        depth_images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            thermal_images.append(b[0])
            depth_images.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])
            difficulties.append(b[4])

        thermal_images = torch.stack(thermal_images, dim=0)
        depth_images = torch.stack(depth_images, dim=0)

        return thermal_images, depth_images, boxes, labels, difficulties   # tensors (N, 1, 300, 300), 3 lists of N tensors each

    def dataset_mean_std(self):
        # thermal mean_std
        mean_thermal = 0
        std_thermal = 0
        tot_img = 0
        for img in self.thermal_images:
            img = Image.open(img, mode='r')
            img = FT.to_tensor(img).type(torch.FloatTensor)
            mean_thermal += img.mean()
            std_thermal += img.std()
            tot_img += 1
        mean_thermal /= tot_img
        std_thermal /= tot_img

        # depth mean_std
        mean_depth = 0
        std_depth = 0
        tot_img = 0
        for img in self.depth_images:
            img = Image.open(img, mode='r')
            img = FT.to_tensor(img).type(torch.FloatTensor)
            mean_depth += img.mean()
            std_depth += img.std()
            tot_img += 1
        mean_depth /= tot_img
        std_depth /= tot_img

        return [mean_thermal, mean_depth], [std_thermal, std_depth]


class DetectDataset(Dataset):
    """
    A pytorch Dataset class to be used in a Pytorch Dataloader to load test examples.
    """
    def __init__(self, data_folder):
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
        self.thermal_images = [os.path.join(data_folder, 'Thermique', img) for img in os.listdir(os.path.join(self.data_folder, 'Thermique'))]
        self.depth_images = [img.replace('Thermique', 'Profondeur').replace('thermal', 'depth') for img in self.thermal_images]

    def __getitem__(self, i):
        # Read image
        thermal_image = Image.open(self.thermal_images[i], mode='r')
        depth_image = Image.open(self.depth_images[i], mode='r')
        original_width = thermal_image.width
        original_height = thermal_image.height

        image_8bit = convert_16bit_to_8bit(self.thermal_images[i])

        # Apply transformation to the 16 bit image
        thermal_image, depth_image, boxes, labels, difficulties = transform(thermal_image,
                                                                               depth_image,
                                                                               boxes=torch.Tensor([[0, 0, 0, 0]]),
                                                                               labels=None,
                                                                               difficulties=None,
                                                                               split='TEST')

        return thermal_image.type('torch.FloatTensor'), depth_image.type('torch.FloatTensor'), image_8bit, original_width, original_height

    def __len__(self):
        return len(self.thermal_images)






