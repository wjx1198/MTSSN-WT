import torch.nn as nn

from models import resnet

# from metrics import batch_pix_accuracy, batch_intersection_union

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

# __all__ = ['BaseNet', 'MultiEvalModule']
# __all__ = ['BaseNet']


class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, norm_layer=None, root='./pretrain_models'):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        # self.mean = mean
        # self.std = std
        # self.crop_size = crop_size
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, norm_layer=norm_layer, root=root)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True,
                                               norm_layer=norm_layer, root=root)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True,
                                               norm_layer=norm_layer, root=root)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        c1 = self.pretrained.relu(x)
        x = self.pretrained.maxpool(c1)
        c2 = self.pretrained.layer1(x)
        c3 = self.pretrained.layer2(c2)
        c4 = self.pretrained.layer3(c3)
        c5 = self.pretrained.layer4(c4)
        return c1, c2, c3, c4, c5
