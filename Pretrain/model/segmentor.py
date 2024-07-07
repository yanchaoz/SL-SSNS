# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch.nn as nn
import torch.nn.functional as F
import torch

from model.basic import conv3dBlock, upsampleBlock
from model.residual import resBlock_pni


class UNet_PNI(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1,
                 out_planes=3,
                 filters=[28, 36, 48, 64, 80],  # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                 upsample_mode='transposeS',  # transposeS, bilinear
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

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

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4] * 2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3] * 2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2] * 2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1] * 2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])],
                                     [int(filters2[0])],
                                     [(1, 5, 5)],
                                     [1],
                                     [(0, 2, 2)],
                                     [True],
                                     [pad_mode],
                                     [''],
                                     [relu_mode],
                                     init_mode,
                                     bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    def forward(self, x):
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

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            x = self.avgpool(center)
            x = torch.flatten(x, 1)
            return x
        else:
            return out

class UNet_PNI_Noskip(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1, 
                    out_planes=3, 
                    filters=[28, 36, 48, 64, 80],    # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                    upsample_mode='bilinear', 
                    decode_ratio=1, 
                    merge_mode='cat', 
                    pad_mode='zero', 
                    bn_mode='async',   # async or sync
                    relu_mode='elu', 
                    init_mode='kaiming_normal', 
                    bn_momentum=0.001, 
                    do_embed=True,
                    if_sigmoid=True,
                    show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_Noskip, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

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

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1,2,2), upsample_mode, init_mode=init_mode)
        self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1,2,2), upsample_mode, init_mode=init_mode)
        self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1,2,2), upsample_mode, init_mode=init_mode)
        self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1,2,2), upsample_mode, init_mode=init_mode)
        self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])], 
                                        [int(filters2[0])], 
                                        [(1, 5, 5)], 
                                        [1], 
                                        [(0, 2, 2)], 
                                        [True], 
                                        [pad_mode], 
                                        [''], 
                                        [relu_mode], 
                                        init_mode, 
                                        bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

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
        down_features = [conv0, conv1, conv2, conv3]

        return center, down_features

    def decoder(self, center):
        up0 = self.up0(center)
        cat0 = self.cat0(up0)
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        cat1 = self.cat1(up1)
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        cat2 = self.cat2(up2)
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        cat3 = self.cat3(up3)
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)
        up_features = [conv4, conv5, conv6, conv7]

        return out, up_features

    def forward(self, x):
        center, down_features = self.encoder(x)
        out, up_features = self.decoder(center)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            center_features = [center]
            return down_features, center_features, up_features, out
        else:
            return out

if __name__ == "__main__":
    import os
    import numpy as np
    from collections import OrderedDict
    from model.encoder import UNet_PNI_encoder

    input = np.random.random((1, 1, 18, 160, 160)).astype(np.float32)
    x = torch.tensor(input)
    encoder = UNet_PNI_encoder(filters=[32, 64, 128, 256, 512])
    model = UNet_PNI(filters=[32, 64, 128, 256, 512], upsample_mode='transposeS', merge_mode='cat')
    ckpt_path = '../models/model-300000.ckpt'
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k.replace('module.', '') if 'module' in k else k
        new_state_dict[name] = v
    pretrained_dict = new_state_dict
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 1. filter out unnecessary keys
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)  # 2. overwrite entries in the existing state dict
    model.load_state_dict(model_dict)

    # print('Load pre-trained model ...')
    # checkpoint = torch.load(ckpt_path)
    # pretrained_dict = checkpoint['model_weights']
    # trained_gpus = 1
    # if_skip = 'False'
    # if trained_gpus > 1:
    #     pretained_model_dict = OrderedDict()
    #     for k, v in pretrained_dict.items():
    #         name = k[7:]  # remove module.
    #         pretained_model_dict[name] = v
    # else:
    #     pretained_model_dict = pretrained_dict
    # from utils.encoder_dict import ENCODER_DICT2, ENCODER_DECODER_DICT2
    # model_dict = model.state_dict()
    # encoder_dict = OrderedDict()
    # if if_skip == 'True':
    #     print('Load the parameters of encoder and decoder!')
    #     encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DECODER_DICT2}
    # else:
    #     print('Load the parameters of encoder!')
    #     encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DICT2}
    # model_dict.update(encoder_dict)
    # model.load_state_dict(model_dict)
    # out = model(x)
    # print(out.shape)
