import torch 
from torch import nn
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
from torch.utils.data import random_split
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler,WeightedRandomSampler
from torch.utils.data import random_split
from torch.utils.data import SequentialSampler

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

from resnet_mouse import _ResNetLayer


class MouseNeuralNetwork(nn.Module):
    def __init__(self, length_single_mouse_traj):
        super(MouseNeuralNetwork, self).__init__()
        self.len_lost_in_mousetraj_single = 2
        self.num_hidden_layer_gru = 25
        self.sizeof_x = (length_single_mouse_traj - self.len_lost_in_mousetraj_single) * self.num_hidden_layer_gru

        # 保持卷积层和BatchNorm
        self.conv0 = nn.Conv1d(1, 8, 5, padding=1)
        self.bn0 = nn.BatchNorm1d(8)
        
        # ResNet层
        self.layer1 = _ResNetLayer(in_channel=8, out_channel=16)
        
        # GRU层
        self.rnn = nn.GRU(
            input_size=16,
            hidden_size=self.num_hidden_layer_gru,
            num_layers=3,
            batch_first=False,
            bidirectional=False
        )
        
        # 完全连接层
        self.fc2 = nn.Linear(self.sizeof_x, 2)
        
        # Dropout层
        self.dr1 = nn.Dropout(0.4)
        
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, input):
        # 初始卷积和批标准化
        x = self.relu(self.bn0(self.conv0(input)))
        
        # ResNet层
        x = self.layer1(x)
        x = self.dr1(x)
        
        # 调整维度для GRU
        x = x.permute(0, 2, 1)
        
        # GRU层
        x, _ = self.rnn(x)
        
        # 展平
        x = x.reshape(x.size(0), -1)
        
        # 最后的线性层
        x = self.fc2(x)
        
        return x



class MouseNeuralNetwork2(nn.Module):
    def __init__(self, length_single_mouse_traj):
        super(MouseNeuralNetwork2, self).__init__()
        self.len_mouse_traj = length_single_mouse_traj
        self.end_channel_of_resnet3 = 64
        self.len_lost_in_mousetraj_all = 2
        self.sizeof_x = (length_single_mouse_traj - self.len_lost_in_mousetraj_all) * self.end_channel_of_resnet3

        # 卷积层和BatchNorm
        self.conv0 = nn.Conv1d(1, 8, 5, padding=1)
        self.bn0 = nn.BatchNorm1d(8)
        
        # ResNet层
        self.layer1 = _ResNetLayer(in_channel=8, out_channel=16)
        self.layer2 = _ResNetLayer(in_channel=16, out_channel=32)
        self.layer3 = _ResNetLayer(in_channel=32, out_channel=64)
        
        # 完全连接层
        self.fc = nn.Linear(self.sizeof_x, 300)
        self.fc2 = nn.Linear(300, 2)
        
        # Dropout层
        self.dr1 = nn.Dropout(0.4)
        
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, input):
        # 初始卷积和批标准化
        x = self.relu(self.bn0(self.conv0(input)))
        
        # ResNet层
        x = self.layer1(x)
        x = self.dr1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dr1(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc(x))
        x = self.fc2(x)
        
        return x