import os
import cv2
import h5py
import math
import random
import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Provider_valid(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_type = cfg.MODEL.model_type
        valid_dataset_name = cfg.DATA.dataset_name
        print('valid dataset:', valid_dataset_name)

        self.crop_size = [18, 160, 160]
        self.net_padding = [0, 0, 0]

        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

        if valid_dataset_name == 'ac3_ac4':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['AC3_inputs.h5']
        else:
            raise AttributeError('No this dataset type!')

        self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)

        # load dataset
        self.dataset = []
        for k in range(len(self.train_datasets)):
            print('load ' + self.train_datasets[k] + ' ...')
            try:
                f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
                data = f_raw['main'][:100]
                f_raw.close()
            except:
                data = imageio.volread(os.path.join(self.folder_name, self.train_datasets[k]))[:100]
            self.dataset.append(data)
        self.origin_data_shape = list(self.dataset[0].shape)

        padding_z = 0
        padding_xy = 0
        num_z = 12
        num_xy = 23
        self.stride = [8, 40, 40]

        self.valid_padding = [padding_z, padding_xy, padding_xy]
        self.num_zyx = [num_z, num_xy, num_xy]
        for k in range(len(self.dataset)):
            self.dataset[k] = np.pad(self.dataset[k], ((self.valid_padding[0], self.valid_padding[0]),
                                                       (self.valid_padding[1], self.valid_padding[1]),
                                                       (self.valid_padding[2], self.valid_padding[2])),
                                     mode='reflect')

        self.raw_data_shape = list(self.dataset[0].shape)

        self.num_per_dataset = self.num_zyx[0] * self.num_zyx[1] * self.num_zyx[2]
        self.iters_num = self.num_per_dataset * len(self.dataset)


def __getitem__(self, index):
    # print(index)
    pos_data = index // self.num_per_dataset
    pre_data = index % self.num_per_dataset
    pos_z = pre_data // (self.num_zyx[1] * self.num_zyx[2])
    pos_xy = pre_data % (self.num_zyx[1] * self.num_zyx[2])
    pos_x = pos_xy // self.num_zyx[2]
    pos_y = pos_xy % self.num_zyx[2]

    fromz = pos_z * self.stride[0]
    endz = fromz + self.crop_size[0]
    if endz > self.raw_data_shape[0]:
        endz = self.raw_data_shape[0]
        fromz = endz - self.crop_size[0]
    fromy = pos_y * self.stride[1]
    endy = fromy + self.crop_size[1]
    if endy > self.raw_data_shape[1]:
        endy = self.raw_data_shape[1]
        fromy = endy - self.crop_size[1]
    fromx = pos_x * self.stride[2]
    endx = fromx + self.crop_size[2]
    if endx > self.raw_data_shape[2]:
        endx = self.raw_data_shape[2]
        fromx = endx - self.crop_size[2]
    self.pos = [fromz, fromx, fromy]

    imgs = self.dataset[pos_data][fromz:endz, fromx:endx, fromy:endy].copy()
    imgs = imgs.astype(np.float32) / 255.0
    imgs = imgs[np.newaxis, ...]
    imgs = np.ascontiguousarray(imgs, dtype=np.float32)

    return imgs


def __len__(self):
    return self.iters_num
