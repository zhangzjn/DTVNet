from torch import nn
import torch

class FlowEncoder(nn.Module):
    def __init__(self, type, ngf, dlatent_size):
        super(FlowEncoder, self).__init__()
        self.downConv1 = nn.Conv3d(2, ngf, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)  
        self.downConv2 = nn.Conv3d(ngf, ngf * 2, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        self.downConv3 = nn.Conv3d(ngf * 2, ngf * 4, 4, 2, 1, bias=False)
        self.downConv4 = nn.Conv3d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.downConv5 = nn.Conv3d(ngf * 8, ngf * 16, 4, 2, 1, bias=False)
        self.downConv6 = nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)

        self.downBN2 = nn.BatchNorm3d(ngf * 2)
        self.downBN3 = nn.BatchNorm3d(ngf * 4)
        self.downBN4 = nn.BatchNorm3d(ngf * 8)
        self.downBN5 = nn.BatchNorm3d(ngf * 16)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.fc1 = nn.Linear(ngf * 16, dlatent_size)

    def forward(self, x):
        downx1 = self.downConv1(x)
        downx2 = self.downConv2(downx1)
        downx2 = self.downBN2(downx2)
        downx2 = self.lrelu(downx2)
        downx3 = self.downConv3(downx2)
        downx3 = self.downBN3(downx3)
        downx3 = self.lrelu(downx3)
        downx4 = self.downConv4(downx3)
        downx4 = self.downBN4(downx4)
        downx4 = self.lrelu(downx4)
        downx5 = self.downConv5(downx4)
        downx5 = self.downBN5(downx5)
        downx5 = self.lrelu(downx5)
        downx6 = self.downConv6(downx5)

        b, s = downx6.size(0), downx6.size(1)
        #print(downx6.shape)
        out = downx6.view(b, s)
        
        out = self.fc1(out)
        return out