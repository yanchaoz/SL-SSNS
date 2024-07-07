from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch.nn as nn
import torch
import sys
sys.path.append('/home/zhangyc/2023_project/pretrain')
from model.basic import conv3dBlock
from model.residual import resBlock_pni


class UNet_PNI_encoder(nn.Module):
    def __init__(self, in_planes=1,
                 filters=[32, 64, 128, 256, 512],
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 num_classes=None):
        super(UNet_PNI_encoder, self).__init__()
        filters2 = filters[:1] + filters
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.num_classes = num_classes

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def encoder(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)
        return center

    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        return x


if __name__ == "__main__":
    import numpy as np

    ENCODER_DICT = [
        'embed_in',
        'conv0',
        'conv1',
        'conv2',
        'conv3',
        'center'
    ]
    input = np.random.random((2, 1, 18, 160, 160)).astype(np.float32)
    x = torch.tensor(input)
    model = UNet_PNI_encoder(filters=[32, 64, 128, 256, 512])
    reaentation = model(x)
    print(reaentation.shape)
