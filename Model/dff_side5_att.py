###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################
from __future__ import division
import torch
import torch.nn as nn
from models.base import BaseNet
import pdb


class DFF(nn.Module):
    r"""Dynamic Feature Fusion for Semantic Edge Detection

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Yuan Hu, Yunpeng Chen, Xiang Li, Jiashi Feng. "Dynamic Feature Fusion
        for Semantic Edge Detection" *IJCAI*, 2019

    """

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(DFF, self).__init__()
        self.nclass = nclass

        # self.ada_learner = LocationAdaptiveLearner(nclass, nclass * 4, nclass * 4, norm_layer=norm_layer)

        self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   norm_layer(1))
        self.side2 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(512, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(2048, nclass, 1, bias=True),
                                   norm_layer(nclass),
                                   nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))
        self.chatten = cSE(nclass)

        # self.side5_w = nn.Sequential(nn.Conv2d(2048, nclass * 4, 1, bias=True),
        #                              norm_layer(nclass * 4),
        #                              nn.ConvTranspose2d(nclass * 4, nclass * 4, 16, stride=8, padding=4, bias=False))
        # self.conv_result = nn.Sequential(nn.Conv2d(24, nclass, 1, bias=True),
        #                            norm_layer(nclass),
        #                            nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))
        # self.conv_result = nn.Conv2d(24, nclass, 1, bias=True)

    def forward(self, c1, c2, c3, c5):
        # c1, c2, c3, c4, c5 = self.base_forward(x)
        side1 = self.side1(c1)  # (N, 1, H, W)
        side2 = self.side2(c2)  # (N, 1, H, W)
        side3 = self.side3(c3)  # (N, 1, H, W)
        side5 = self.side5(c5)  # (N, NUM_CLASS, H, W)
        # side5_w = self.side5_w(c5)  # (N, NUM_CLASS*4, H, W)

        # ada_weights = self.ada_learner(side5_w)  # (N, NUM_CLASS, 4, H, W)

        slice5 = side5[:, 0:1, :, :]  # (N, 1, H, W)
        fuse = torch.cat((slice5, side1, side2, side3), 1)
        for i in range(side5.size(1) - 1):
            slice5 = side5[:, i + 1:i + 2, :, :]  # (N, 1, H, W)
            fuse = torch.cat((fuse, slice5, side1, side2, side3), 1)  # (N, NUM_CLASS*4, H, W)

        fuse = fuse.view(fuse.size(0), self.nclass, -1, fuse.size(2), fuse.size(3))  # (N, NUM_CLASS, 4, H, W)
        # fuse = self.chatten(fuse)
        # fuse = self.conv_result(fuse)
        # fuse = torch.mul(fuse, ada_weights)  # (N, NUM_CLASS, 4, H, W)
        fuse = torch.sum(fuse, 2)  # (N, NUM_CLASS, H, W)
        side5 = self.chatten(side5)

        outputs = [fuse, side5]

        return tuple(outputs)

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2]
        z = self.Conv_Excitation(z)  # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class LocationAdaptiveLearner(nn.Module):
    """docstring for LocationAdaptiveLearner"""

    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(LocationAdaptiveLearner, self).__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels))

    def forward(self, x):
        # x:side5_w (N, 19*4, H, W)
        x = self.conv1(x)  # (N, 19*4, H, W)
        x = self.conv2(x)  # (N, 19*4, H, W)
        x = self.conv3(x)  # (N, 19*4, H, W)
        x = x.view(x.size(0), self.nclass, -1, x.size(2), x.size(3))  # (N, NUM_CLASS, 4, H, W)
        return x


class DFF_ori(BaseNet):
    r"""Dynamic Feature Fusion for Semantic Edge Detection

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Yuan Hu, Yunpeng Chen, Xiang Li, Jiashi Feng. "Dynamic Feature Fusion
        for Semantic Edge Detection" *IJCAI*, 2019

    """

    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DFF_ori, self).__init__(nclass, backbone, norm_layer=norm_layer, **kwargs)
        self.nclass = nclass

        self.ada_learner = LocationAdaptiveLearner(nclass, nclass * 4, nclass * 4, norm_layer=norm_layer)

        self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   norm_layer(1))
        self.side2 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(512, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(2048, nclass, 1, bias=True),
                                   norm_layer(nclass),
                                   nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))

        self.side5_w = nn.Sequential(nn.Conv2d(2048, nclass * 4, 1, bias=True),
                                     norm_layer(nclass * 4),
                                     nn.ConvTranspose2d(nclass * 4, nclass * 4, 16, stride=8, padding=4, bias=False))

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.base_forward(x)

        side1 = self.side1(c1)  # (N, 1, H, W)
        side2 = self.side2(c2)  # (N, 1, H, W)
        side3 = self.side3(c3)  # (N, 1, H, W)
        side5 = self.side5(c5)  # (N, NUM_CLASS, H, W)
        side5_w = self.side5_w(c5)  # (N, NUM_CLASS*4, H, W)

        ada_weights = self.ada_learner(side5_w)  # (N, NUM_CLASS, 4, H, W)

        slice5 = side5[:, 0:1, :, :]  # (N, 1, H, W)
        fuse = torch.cat((slice5, side1, side2, side3), 1)
        for i in range(side5.size(1) - 1):
            slice5 = side5[:, i + 1:i + 2, :, :]  # (N, 1, H, W)
            fuse = torch.cat((fuse, slice5, side1, side2, side3), 1)  # (N, NUM_CLASS*4, H, W)

        fuse = fuse.view(fuse.size(0), self.nclass, -1, fuse.size(2), fuse.size(3))  # (N, NUM_CLASS, 4, H, W)
        fuse = torch.mul(fuse, ada_weights)  # (N, NUM_CLASS, 4, H, W)
        fuse = torch.sum(fuse, 2)  # (N, NUM_CLASS, H, W)

        outputs = [side5, fuse]

        return tuple(outputs)


def get_dff(nclass, backbone='resnet50', root='./pretrain_models', **kwargs):
    r"""DFF model from the paper "Dynamic Feature Fusion for Semantic Edge Detection"
    """
    # acronyms = {
    #     'cityscapes': 'cityscapes',
    #     'sbd': 'sbd',
    # }
    # infer number of classes
    from WTCNN_Edge import EdgeDetection
    model = DFF_ori(nclass, backbone=backbone, root=root, **kwargs)
    # if pretrained:
    #     from .model_store import get_model_file
    #     model.load_state_dict(torch.load(
    #         get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
    #         strict=False)
    return model
