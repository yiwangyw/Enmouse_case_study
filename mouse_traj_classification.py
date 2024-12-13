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
    def __init__(self, length_single_mouse_traj, num_features=11):
        super(MouseNeuralNetwork, self).__init__()
        self.len_lost_in_mousetraj_single = 2
        self.num_hidden_layer_gru = 25
        self.sizeof_x = (length_single_mouse_traj - self.len_lost_in_mousetraj_single) * self.num_hidden_layer_gru

        # 修改第一个卷积层的输入通道数为特征数量
        self.conv0 = nn.Conv1d(num_features, 8, 5, padding=1)
        self.bn0 = nn.BatchNorm1d(8)
        
        self.layer1 = _ResNetLayer(in_channel=8, out_channel=16)
        self.layer2 = _ResNetLayer(in_channel=16, out_channel=32)
        self.layer3 = _ResNetLayer(in_channel=32, out_channel=64)
        self.layer4 = _ResNetLayer(in_channel=64, out_channel=128)
        
        self.fc = nn.Linear(40, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.sizeof_x, 2)
        self.dr1 = nn.Dropout(0.2)
        self.dr2 = nn.Dropout(0.25)
        self.rnn = nn.GRU(32, self.num_hidden_layer_gru, 3, False, True, bidirectional=False)
        self.sig = nn.Sigmoid()
        self.flatten = nn.Flatten(1)

    def forward(self, input):
        # input shape: (batch_size, num_features, sequence_length)
        x = self.relu(self.bn0(self.conv0(input)))
        x = self.layer1(x)
        x = self.dr1(x)
        x = self.layer2(x)
        x = self.dr1(x)
        x = x.permute(0, 2, 1)
        x, self.hidden = self.rnn(x)
        x = x.reshape((x.size(0), -1))
        x = self.fc2(x)
        return x


class MouseNeuralNetwork2(nn.Module):
    def __init__(self, length_single_mouse_traj):
        super(MouseNeuralNetwork2, self).__init__()
        self.len_mouse_traj = length_single_mouse_traj
        # 修改输入通道数为11（特征数量）
        self.conv0 = nn.Conv1d(11, 8, 5, padding=1)  # 修改输入通道数
        self.bn0 = nn.BatchNorm1d(8)
        self.end_channel_of_resnet3 = 64
        
        self.layer1 = _ResNetLayer(in_channel=8, out_channel=16)
        self.layer2 = _ResNetLayer(in_channel=16, out_channel=32)
        self.layer3 = _ResNetLayer(in_channel=32, out_channel=64)
        
        self.len_lost_in_mousetraj_all = 2
        self.sizeof_x = (length_single_mouse_traj - self.len_lost_in_mousetraj_all) * self.end_channel_of_resnet3
        
        self.fc = nn.Linear(self.sizeof_x, 300)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(300, 2)
        self.dr1 = nn.Dropout(0.2)
        self.dr2 = nn.Dropout(0.25)

    def forward(self, input):
        # input shape: (batch_size, num_features, sequence_length)
        x = self.relu(self.bn0(self.conv0(input)))
        x = self.layer1(x)
        x = self.dr1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dr1(x)
        x = x.view((x.size(0), -1))
        x = self.fc(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x