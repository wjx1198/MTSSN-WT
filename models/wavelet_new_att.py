# Import Statements
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import pdb
from models.base import BaseNet
from models.wavelet import wavelet_transform


# torch.nn.Module.dump_patches = True

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制


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


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse + U_sse


class waveletCNN(nn.Module):

    def __init__(self, nclass, wvlt_transform):
        # def __init__(self):

        super(waveletCNN, self).__init__()

        self.wvlt_transform = wvlt_transform
        # self.wvlt_transform = wavelet_transform

        ### Main Convolutions ###
        # Define the k1 set of convolutions
        self.convk1a = nn.Conv2d(in_channels=67, out_channels=64, kernel_size=1, padding=0)
        self.convk1a_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.convk1a.weight)
        self.convk1b = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.convk1b_normed = nn.BatchNorm2d(128)
        torch_init.xavier_normal_(self.convk1b.weight)

        # Define the k2 set of convolutions
        self.convk2a = nn.Conv2d(in_channels=388, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.convk2a_normed = nn.BatchNorm2d(256)
        torch_init.xavier_normal_(self.convk2a.weight)
        self.convk2b = nn.Conv2d(in_channels=784, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.convk2b_normed = nn.BatchNorm2d(512)
        torch_init.xavier_normal_(self.convk2b.weight)
        self.convk2c = nn.Conv2d(in_channels=1600, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.convk2c_normed = nn.BatchNorm2d(512)
        torch_init.xavier_normal_(self.convk2c.weight)

        ###Self Attention Mechanism
        self.attention = scSE(in_channels=768)

        ### Invconnections after wavelet transforms ###
        self.invconvs1a = nn.ConvTranspose2d(in_channels=768, out_channels=512, kernel_size=3, padding=1,
                                             output_padding=1, stride=2)#attention_map:768
        self.invconvs1a_normed = nn.BatchNorm2d(512)
        torch_init.xavier_normal_(self.invconvs1a.weight)
        self.invconvs1b = nn.ConvTranspose2d(in_channels=1600, out_channels=256, kernel_size=3, padding=1,
                                             output_padding=1, stride=2)
        self.invconvs1b_normed = nn.BatchNorm2d(256)
        torch_init.xavier_normal_(self.invconvs1b.weight)

        self.invconvs2a = nn.ConvTranspose2d(in_channels=784, out_channels=128, kernel_size=3, padding=1,
                                             output_padding=1, stride=2)
        self.invconvs2a_normed = nn.BatchNorm2d(128)
        torch_init.xavier_normal_(self.invconvs2a.weight)
        self.invconvs2b = nn.ConvTranspose2d(in_channels=388, out_channels=64, kernel_size=3, padding=1,
                                             output_padding=1, stride=2)
        self.invconvs2b_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.invconvs2b.weight)

        ### Fusion Module ###
        self.convk1f = nn.Conv2d(in_channels=131, out_channels=nclass, kernel_size=1, padding=0)

        ### Secondary Convolution ###
        self.convk1s = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, padding=0)
        self.convk1s_normed = nn.BatchNorm2d(4)
        torch_init.xavier_normal_(self.convk1s.weight)
        self.convk2s = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, padding=0)
        self.convk2s_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.convk2s.weight)
        self.convk3s = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, padding=0)
        self.convk3s_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.convk3s.weight)
        self.convk4s = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=1, padding=0)
        self.convk4s_normed = nn.BatchNorm2d(256)
        torch_init.xavier_normal_(self.convk4s.weight)
        self.convk5s = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, padding=0)
        self.convk5s_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.convk5s.weight)
        self.convk6s = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, padding=0)
        self.convk6s_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.convk6s.weight)
        self.convk7s = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, padding=0)
        self.convk7s_normed = nn.BatchNorm2d(4)
        torch_init.xavier_normal_(self.convk7s.weight)
        # self.endaspp = ASPP()

    def forward(self, batch, c1, c2, c3, c4):
        # c1, c2, c3, c4, c5 = self.base_forward(batch)
        # Compute first wavelet transform
        wvlt1_batch = self.wvlt_transform.batch_transform(batch)
        # wvlt1_batch_new = wvlt1_batch[0, :, :, :].unsqueeze(0)
        if wvlt1_batch.shape[0] == 1:
            wvlt1_batch_new = wvlt1_batch[:, 0, :, :, :].squeeze().unsqueeze(0)
        else:
            wvlt1_batch_new = wvlt1_batch[:, 0, :, :, :].squeeze()
        # Apply the second wavelet transform
        wvlt2_batch = self.wvlt_transform.batch_transform(wvlt1_batch_new)
        # wvlt2_batch_new = wvlt2_batch[0, :, :, :].unsqueeze(0)
        if wvlt2_batch.shape[0] == 1:
            wvlt2_batch_new = wvlt2_batch[:, 0, :, :, :].squeeze().unsqueeze(0)
        else:
            wvlt2_batch_new = wvlt2_batch[:, 0, :, :, :].squeeze()
        # Apply the third wavelet transform
        wvlt3_batch = self.wvlt_transform.batch_transform(wvlt2_batch_new)
        # wvlt3_batch_new = wvlt3_batch[0, :, :, :].unsqueeze(0)
        if wvlt3_batch.shape[0] == 1:
            wvlt3_batch_new = wvlt3_batch[:, 0, :, :, :].squeeze().unsqueeze(0)
        else:
            wvlt3_batch_new = wvlt3_batch[:, 0, :, :, :].squeeze()
        # Apply the fourth wavelet transform
        wvlt4_batch = self.wvlt_transform.batch_transform(wvlt3_batch_new)
        # wvlt4_batch_new = wvlt4_batch[0, :, :, :].unsqueeze(0)
        if wvlt4_batch.shape[0] == 1:
            wvlt4_batch_new = wvlt4_batch[:, 0, :, :, :].squeeze().unsqueeze(0)
        else:
            wvlt4_batch_new = wvlt4_batch[:, 0, :, :, :].squeeze()

        # Apply the k1 convolution layers
        batch = torch.cat([c1, batch], dim=1)
        cvk1a_batch = func.relu(self.convk1a_normed(self.convk1a(batch)))
        cvk1b_batch = func.relu(self.convk1b_normed(self.convk1b(cvk1a_batch)))

        # Apply the s1 skip connection
        cvs1_batch = func.relu(self.convk1s_normed(self.convk1s(wvlt1_batch_new)))
        cvs2_batch = func.relu(self.convk2s_normed(self.convk2s(wvlt2_batch_new)))
        cvs3_batch = func.relu(self.convk3s_normed(self.convk3s(wvlt3_batch_new)))
        cvs4_batch = func.relu(self.convk4s_normed(self.convk4s(wvlt4_batch_new)))
        cvs5_batch = func.relu(self.convk5s_normed(self.convk5s(wvlt3_batch_new)))
        cvs6_batch = func.relu(self.convk6s_normed(self.convk6s(wvlt2_batch_new)))
        cvs7_batch = func.relu(self.convk7s_normed(self.convk7s(wvlt1_batch_new)))
        # Concatenate output
        cvk1_dense_batch = torch.cat([c2, cvk1b_batch, cvs1_batch], dim=1)

        # Apply the k2 convolution layers
        cvk2a_batch = func.relu(self.convk2a_normed(self.convk2a(cvk1_dense_batch)))
        cvk2_dense_batch = torch.cat([c3, cvk2a_batch, cvs2_batch], dim=1)
        cvk2b_batch = func.relu(self.convk2b_normed(self.convk2b(cvk2_dense_batch)))
        cvk3_dense_batch = torch.cat([c4, cvk2b_batch, cvs3_batch], dim=1)
        cvk2c_batch = func.relu(self.convk2c_normed(self.convk2c(cvk3_dense_batch)))
        cvk4_dense_batch = torch.cat([cvk2c_batch, cvs4_batch], dim=1)

        # Apply Attention
        attention_map = self.attention(cvk4_dense_batch)

        # Apply invconnections after wavelet transforms
        # invcvk1a_batch = func.relu(self.invconvs1a_normed(self.invconvs1a(cvk4_dense_batch)))
        invcvk1a_batch = func.relu(self.invconvs1a_normed(self.invconvs1a(attention_map)))
        invcvk1_dense_batch = torch.cat([c4, invcvk1a_batch, cvs5_batch], dim=1)
        invcvk1b_batch = func.relu(self.invconvs1b_normed(self.invconvs1b(invcvk1_dense_batch)))
        invcvk2_dense_batch = torch.cat([c3, invcvk1b_batch, cvs6_batch], dim=1)
        invcvk2a_batch = func.relu(self.invconvs2a_normed(self.invconvs2a(invcvk2_dense_batch)))
        invcvk3_dense_batch = torch.cat([c2, invcvk2a_batch, cvs7_batch], dim=1)
        invcvk2b_batch = func.relu(self.invconvs2b_normed(self.invconvs2b(invcvk3_dense_batch)))
        invcvk4_dense_batch = torch.cat([invcvk2b_batch, batch], dim=1)
        # aspp_output = self.endaspp(invcvk4_dense_batch)

        # Apply Fusion Module
        output_batch = self.convk1f(invcvk4_dense_batch)
        # output_batch = self.convk1f(aspp_output)

        return output_batch

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]

        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features


class baselineCNN(nn.Module):

    def __init__(self, num_class=6):
        super(baselineCNN, self).__init__()

        # Define the k1 set of convolutions
        self.convk1a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, padding=0)
        self.convk1a_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.convk1a.weight)
        self.convk1b = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.convk1b_normed = nn.BatchNorm2d(128)
        torch_init.xavier_normal_(self.convk1b.weight)

        # Define the k2 set of convolutions
        self.convk2a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.convk2a_normed = nn.BatchNorm2d(256)
        torch_init.xavier_normal_(self.convk2a.weight)
        self.convk2b = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.convk2b_normed = nn.BatchNorm2d(512)
        torch_init.xavier_normal_(self.convk2b.weight)
        self.convk2c = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.convk2c_normed = nn.BatchNorm2d(512)
        torch_init.xavier_normal_(self.convk2c.weight)

        ### Invconnections after wavelet transforms ###
        self.invconvs1a = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, padding=1,
                                             output_padding=1, stride=2)
        self.invconvs1a_normed = nn.BatchNorm2d(512)
        torch_init.xavier_normal_(self.invconvs1a.weight)
        self.invconvs1b = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1,
                                             output_padding=1, stride=2)
        self.invconvs1b_normed = nn.BatchNorm2d(256)
        torch_init.xavier_normal_(self.invconvs1b.weight)

        self.invconvs2a = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1,
                                             output_padding=1, stride=2)
        self.invconvs2a_normed = nn.BatchNorm2d(128)
        torch_init.xavier_normal_(self.invconvs2a.weight)
        self.invconvs2b = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1,
                                             output_padding=1, stride=2)
        self.invconvs2b_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.invconvs2b.weight)

        ### Output Module ###
        self.convk1f = nn.Conv2d(in_channels=64, out_channels=num_class, kernel_size=1, padding=0)

    def forward(self, batch):
        # Apply the k1 convolution layers
        cvk1a_batch = func.relu(self.convk1a_normed(self.convk1a(batch)))
        cvk1b_batch = func.relu(self.convk1b_normed(self.convk1b(cvk1a_batch)))

        # Apply the k2 convolution layers
        cvk2a_batch = func.relu(self.convk2a_normed(self.convk2a(cvk1b_batch)))
        cvk2b_batch = func.relu(self.convk2b_normed(self.convk2b(cvk2a_batch)))
        cvk2c_batch = func.relu(self.convk2c_normed(self.convk2c(cvk2b_batch)))

        # Apply invconnections after wavelet transforms
        invcvk1a_batch = func.relu(self.invconvs1a_normed(self.invconvs1a(cvk2c_batch)))
        invcvk1b_batch = func.relu(self.invconvs1b_normed(self.invconvs1b(invcvk1a_batch)))
        invcvk2a_batch = func.relu(self.invconvs2a_normed(self.invconvs2a(invcvk1b_batch)))
        invcvk2b_batch = func.relu(self.invconvs2b_normed(self.invconvs2b(invcvk2a_batch)))

        # Apply Output Module
        output_batch = self.convk1f(invcvk2b_batch)

        return output_batch

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]

        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features
