import torch.nn as nn
from models.base import BaseNet
from models.wavelet import wavelet_transform
from models.dff_side5_att import DFF
from models.wavelet_new_att import waveletCNN


class Fusion(BaseNet):
    def __init__(self,  nclass, backbone, wvlt_transform, computer_device, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Fusion, self).__init__(nclass, backbone, norm_layer, **kwargs)

        # self.BaseNet = BaseNet(nclass=nclass, backbone=backbone, norm_layer=norm_layer)
        self.wavelet_transform = wavelet_transform(wvlt_transform, computer_device)
        self.wavelet = waveletCNN(wvlt_transform=self.wavelet_transform, nclass=nclass)
        self.dff = DFF(nclass=nclass, norm_layer=norm_layer)

    def forward(self, batch):
        c1, c2, c3, c4, c5 = self.base_forward(batch)
        output = self.wavelet(batch, c1, c2, c3, c4)
        output_edge = self.dff(c1, c2, c3, c5)

        return output, output_edge

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]

        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features




