import torch 
from torch import nn
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
from torchvision.datasets import mnist # 获取数据集
from torch.utils.data import random_split
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler,WeightedRandomSampler
from torch.utils.data import random_split
from torch.utils.data import SequentialSampler

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import random
class _ResNetLayer(nn.Module):

    def __init__(self,  in_channel , out_channel, stride=1,downsample=None, bn=False):
        super(_ResNetLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, 5,padding=2)  # （8，16,1）    50-2     （16,32,3）    36-2    (32,64,3)       22-2
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，16,1）   48-2     （32,64,3）    34-2    (64,128,3)      20-2
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.conv3 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，16,1）   46-2     （64,128,3）   32-2     (128,256,3)     18-2
        self.bn3 = nn.BatchNorm1d(out_channel)
        self.conv4 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，16,3）  44-2      （128,256,3）  30-2    (256,512,3)      16-2
        self.bn4 = nn.BatchNorm1d(out_channel)
        self.conv5 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，256,3）  42-2     （256,512,3）   28-2     (512,1024,3)    14-2
        self.bn5 = nn.BatchNorm1d(out_channel)
        self.conv6 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，512,3）  40-2     （512,1024,3）   26-2     (1024,2048,3)   12-2
        self.bn6 = nn.BatchNorm1d(out_channel)
        self.conv7 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，1024,3）  38-2     （1024,2048,3）   24-2    (2048,4096,3)   10-2
        self.bn7 = nn.BatchNorm1d(out_channel)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channel)
        )
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.conv7(out)
        out = self.bn7(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        # print("x shape after residual:", residual.shape)
        # print("x shape after out:", out.shape)
        out = out + residual
        out = self.relu(out)
        return out


class _ResNetLayer2(nn.Module):
    """
    ResNetLayer Module for building ResNet
    """

    def __init__(self,  in_channel , out_channel, stride=1,downsample=None, bn=False):
        super(_ResNetLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, 5,padding=2)  # （8，16,1）    50-2     （16,32,3）    36-2    (32,64,3)       22-2
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，16,1）   48-2     （32,64,3）    34-2    (64,128,3)      20-2
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.conv3 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，16,1）   46-2     （64,128,3）   32-2     (128,256,3)     18-2
        self.bn3 = nn.BatchNorm1d(out_channel)
        self.conv4 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，16,3）  44-2      （128,256,3）  30-2    (256,512,3)      16-2
        self.bn4 = nn.BatchNorm1d(out_channel)
        self.conv5 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，256,3）  42-2     （256,512,3）   28-2     (512,1024,3)    14-2
        self.bn5 = nn.BatchNorm1d(out_channel)
        self.conv6 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，512,3）  40-2     （512,1024,3）   26-2     (1024,2048,3)   12-2
        self.bn6 = nn.BatchNorm1d(out_channel)
        self.conv7 = nn.Conv1d(out_channel, out_channel, 5,padding=2)  # （16，1024,3）  38-2     （1024,2048,3）   24-2    (2048,4096,3)   10-2
        self.bn7 = nn.BatchNorm1d(out_channel)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channel)
        )
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.conv7(out)
        out = self.bn7(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = out + residual
        out = self.relu(out)
        return out