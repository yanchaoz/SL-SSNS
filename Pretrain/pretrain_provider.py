from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import h5py
import time
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.augmentation import SimpleAugment
from utils.consistency_aug_perturbations import Intensity
from utils.consistency_aug_perturbations import GaussBlur
from utils.consistency_aug_perturbations import GaussNoise
from utils.consistency_aug_perturbations import Cutout
from utils.consistency_aug_perturbations import Elastic
from utils.augmentation import SimpleAugment as Filp
import imageio


class Train(Dataset):
    def __init__(self, cfg):
        super(Train, self).__init__()
        self.cfg = cfg
        self.model_type = cfg.MODEL.model_type

        self.crop_size = [18, 160, 160]
        self.patch_size = [18, 160, 160]
        self.net_padding = [0, 0, 0]

        if cfg.DATA.unlabel_dataset == 'ac3_ac4':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['AC3_inputs.h5', 'AC4_inputs.h5']
        else:
            raise AttributeError('No this dataset type!')

        self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
        self.simple_aug = SimpleAugment()
        self.min_kernel_size = cfg.DATA.min_kernel_size
        self.max_kernel_size = cfg.DATA.max_kernel_size
        self.min_sigma = cfg.DATA.min_sigma
        self.max_sigma = cfg.DATA.max_sigma
        self.min_noise_std = cfg.DATA.min_noise_std
        self.max_noise_std = cfg.DATA.max_noise_std
        # load dataset
        self.dataset = []
        for k in range(len(self.train_datasets)):
            print('load ' + self.train_datasets[k] + ' ...')
            try:
                f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
                data = f_raw['main'][:]
                f_raw.close()
            except:
                data = imageio.volread(os.path.join(self.folder_name, self.train_datasets[k]))
            self.dataset.append(data)

        self.raw_data_shape = list(self.dataset[0].shape)
        self.crop_from_origin = [self.crop_size[i] for i in range(len(self.crop_size))]
        self.perturbations_init()

    def __getitem__(self, index):
        k = random.randint(0, len(self.train_datasets) - 1)
        used_data = self.dataset[k]
        used_data_shape = used_data.shape
        random_z = random.randint(0, used_data_shape[0] - self.crop_from_origin[0])
        random_y = random.randint(0, used_data_shape[1] - self.crop_from_origin[1])
        random_x = random.randint(0, used_data_shape[2] - self.crop_from_origin[2])

        imgs = used_data[random_z:random_z + self.crop_from_origin[0], \
               random_y:random_y + self.crop_from_origin[1], \
               random_x:random_x + self.crop_from_origin[2]].copy()

        imgs = imgs.astype(np.float32) / 255.0
        [imgs] = self.simple_aug([imgs])
        imgs_aug1 = imgs.copy()
        imgs_aug1 = self.apply_perturbations(imgs_aug1)
        imgs_aug2 = imgs.copy()
        imgs_aug2 = self.apply_perturbations(imgs_aug2)

        sub_z1, sub_x1, sub_y1 = random.randint(0, 1), random.randint(0, 10), random.randint(0, 10)
        if random.random() > 1.0:
            sub_z2, sub_x2, sub_y2 = random.randint(0, 1), random.randint(0, 10), random.randint(0, 10)
        else:
            sub_z2, sub_x2, sub_y2 = sub_z1, sub_x1, sub_y1

        imgs_aug1 = imgs_aug1[sub_z1:sub_z1 + self.patch_size[0], sub_x1:sub_x1 + self.patch_size[1],
                    sub_y1:sub_y1 + self.patch_size[2]]
        imgs_aug2 = imgs_aug2[sub_z2:sub_z2 + self.patch_size[0], sub_x2:sub_x2 + self.patch_size[1],
                    sub_y2:sub_y2 + self.patch_size[2]]

        imgs_aug1 = imgs_aug1[np.newaxis, ...]
        imgs_aug1 = np.ascontiguousarray(imgs_aug1, dtype=np.float32)

        imgs_aug2 = imgs_aug2[np.newaxis, ...]
        imgs_aug2 = np.ascontiguousarray(imgs_aug2, dtype=np.float32)
        return imgs_aug1, imgs_aug2

    def perturbations_init(self):

        self.per_intensity = Intensity()
        self.per_gaussnoise = GaussNoise(min_std=self.min_noise_std, max_std=self.max_noise_std, norm_mode='trunc')
        self.per_gaussblur = GaussBlur(min_kernel=self.min_kernel_size, max_kernel=self.max_kernel_size,
                                       min_sigma=self.min_sigma, max_sigma=self.max_sigma)
        self.per_cutout = Cutout(model_type=self.model_type)

        self.per_misalign = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 0, 0], prob_slip=0.2,
                                    prob_shift=0.2, max_misalign=17, padding=20)
        self.per_elastic = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 2, 2], padding=20)

    def apply_perturbations(self, data):

        rand_per = random.randint(0, 6)
        if rand_per == 0:
            data = self.per_intensity(data)
        if rand_per == 1:
            data = self.per_gaussnoise(data)
        if rand_per == 2:
            data = self.per_gaussblur(data)
        if rand_per == 3:
            data = self.per_cutout(data)
        if rand_per == 4:
            data = self.per_misalign(data)
        if rand_per == 5:
            data = self.per_elastic(data)
        if rand_per == 6:
            [data] = self.simple_aug([data])
        return data

    def __len__(self):
        return int(sys.maxsize)


class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        self.data = Train(cfg)
        self.batch_size = cfg.TRAIN.batch_size
        self.num_workers = cfg.TRAIN.num_workers
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.data.num_per_epoch

    def build(self):
        self.data_iter = iter(
            DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                       shuffle=False, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch
