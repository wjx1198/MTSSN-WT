from __future__ import print_function, division

import datetime
import os
import sys

import cv2
import numpy as np
import utils.utils as U
import torch
import torch.utils.data as data
import pdb


class EdgeDetection(data.Dataset):
    NUM_CLASS = 6

    def __init__(self, split, root='./road_dataset_edge_new'):
        super(EdgeDetection, self).__init__()
        self.split = split
        self.images, self.masks, self.masks_edge = prepare_data(root, self.split)
        if split != 'vis':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, idx):
        paths = U.filepath_to_name(self.images[idx])
        img = U.load_image1(self.images[idx])
        mask = U.load_image2(self.masks[idx])
        mask_edge = U.load_image2(self.masks_edge[idx])
        img, _ = U.resize_data(img, img, 512, 512)
        mask, _ = U.resize_data(mask, mask, 512, 512)
        img = np.float32(img) / 255.0
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        mask_edge = np.transpose(mask_edge, (2, 0, 1))
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        mask_edge = torch.from_numpy(mask_edge)
        return img, mask, mask_edge, paths

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        return self.NUM_CLASS


def prepare_data(dataset_dir, mode):
    train_input_names = []
    train_output_names = []
    train_edge_names = []
    val_input_names = []
    val_output_names = []
    val_edge_names = []
    test_input_names = []
    test_output_names = []
    test_edge_names = []
    if mode == 'train':
        for file in os.listdir(dataset_dir + "/train"):
            # cwd = os.getcwd()
            train_input_names.append(dataset_dir + "/train/" + file)
        for file in os.listdir(dataset_dir + "/train_labels"):
        #     # cwd = os.getcwd()
            train_output_names.append(dataset_dir + "/train_labels/" + file)
        for file in os.listdir(dataset_dir + "/train_sedge_labels"):
            # cwd = os.getcwd()
            train_edge_names.append(dataset_dir + "/train_sedge_labels/" + file)

        train_input_names.sort(), train_output_names.sort(), train_edge_names.sort()
        return train_input_names, train_output_names, train_edge_names

    if mode == 'val':
        for file in os.listdir(dataset_dir + "/val"):
            # cwd = os.getcwd()
            val_input_names.append(dataset_dir + "/val/" + file)
        for file in os.listdir(dataset_dir + "/val_labels"):
        #     cwd = os.getcwd()
            val_output_names.append(dataset_dir + "/val_labels/" + file)
        for file in os.listdir(dataset_dir + "/val_sedge_labels"):
            # cwd = os.getcwd()
            val_edge_names.append(dataset_dir + "/val_sedge_labels/" + file)

        val_input_names.sort(), val_output_names.sort(), val_edge_names.sort()
        return val_input_names, val_output_names, val_edge_names

    if mode == 'test':
        for file in os.listdir(dataset_dir + "/test"):
            # cwd = os.getcwd()
            test_input_names.append(dataset_dir + "/test/" + file)
        for file in os.listdir(dataset_dir + "/test_labels"):
        #     # cwd = os.getcwd()
            test_output_names.append(dataset_dir + "/test_labels/" + file)
        for file in os.listdir(dataset_dir + "/test_sedge_labels"):
            # cwd = os.getcwd()
            test_edge_names.append(dataset_dir + "/test_sedge_labels/" + file)

        test_input_names.sort(), test_output_names.sort(), test_edge_names.sort()
        return test_input_names, test_output_names, test_edge_names
