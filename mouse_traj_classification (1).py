# mouse_traj_classification.py

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, WeightedRandomSampler, SequentialSampler, random_split
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

# 导入 ResNet1D
from resnet_mouse import resnet50_1d

class MouseNeuralNetwork(nn.Module):
    def __init__(self, length_single_mouse_traj, num_features=11):
        super(MouseNeuralNetwork, self).__init__()
        self.len_lost_in_mousetraj_single = 2
        self.num_hidden_layer_gru = 25
        self.sizeof_x = self.num_hidden_layer_gru * self.len_lost_in_mousetraj_single  # 25 * 2 = 50

        # 使用 ResNet1D 作为特征提取器
        self.resnet = resnet50_1d(feature_extractor=True)  # 特征提取模式

        # GRU 和全连接层
        self.gru = nn.GRU(input_size=512, hidden_size=self.num_hidden_layer_gru, num_layers=3, 
                          batch_first=True, bidirectional=False)
        self.fc2 = nn.Linear(self.sizeof_x, 2)  # 二分类输出
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.flatten = nn.Flatten(1)

    def forward(self, input):
        # input shape: (batch_size, num_features=11, sequence_length)
        # 通过 ResNet1D 提取特征
        features = self.resnet(input)  # [batch, 512]

        # 将特征扩展为序列长度为 self.len_lost_in_mousetraj_single
        x = features.unsqueeze(1).repeat(1, self.len_lost_in_mousetraj_single, 1)  # [batch, 2, 512]

        # 通过 GRU
        x, self.hidden = self.gru(x)  # x: [batch, 2, 25]
        x = x.reshape((x.size(0), -1))  # [batch, 50]

        # 全连接层
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, 2]
        return x

class MouseNeuralNetwork2(nn.Module):
    def __init__(self, length_single_mouse_traj):
        super(MouseNeuralNetwork2, self).__init__()
        self.len_mouse_traj = length_single_mouse_traj
        self.end_channel_of_resnet3 = 512  # ResNet1D 特征维度

        self.len_lost_in_mousetraj_all = 2
        self.sizeof_x = self.end_channel_of_resnet3 * self.len_lost_in_mousetraj_all  # 512 * 2 = 1024

        # 使用 ResNet1D 作为特征提取器
        self.resnet = resnet50_1d(feature_extractor=True)  # 特征提取模式

        self.fc = nn.Linear(self.sizeof_x, 300)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(300, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, input):
        # input shape: (batch_size, num_features=11, sequence_length)
        # 通过 ResNet1D 提取特征
        features = self.resnet(input)  # [batch, 512]

        # 重复特征向量
        x = features.repeat(1, self.len_lost_in_mousetraj_all)  # [batch, 512 * 2 = 1024]

        # 全连接层
        x = self.fc(x)  # [batch, 300]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, 2]
        x = F.softmax(x, dim=1)  # 二分类
        return x
