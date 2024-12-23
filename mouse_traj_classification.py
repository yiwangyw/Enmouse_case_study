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

        # 使用 ResNet1D 作为特征提取器
        self.resnet = resnet50_1d(feature_extractor=True)  # 特征提取模式

        # GRU 和全连接层
        self.gru = nn.GRU(input_size=512, hidden_size=self.num_hidden_layer_gru, num_layers=3, 
                          batch_first=True, bidirectional=False)
        self.fc2 = nn.Linear(self.num_hidden_layer_gru, 2)


    def forward(self, input):
        """
        前向传播

        Args:
            input: 张量，形状为 [batch_size, num_features=11, sequence_length]

        Returns:
            输出分类结果，形状为 [batch_size, 2]
        """
        # 通过 ResNet1D 提取特征
        x = self.resnet(input)  # [batch, 512, seq_len_reduced]

        # 调整维度以匹配 GRU 的输入要求
        x = x.permute(0, 2, 1)  # [batch, seq_len_reduced, 512]

        # 通过 GRU
        x, _ = self.gru(x)  # x: [batch, seq_len_reduced, 25]

        # 使用 GRU 最后一个时间步的输出进行分类
        x = x[:, -1, :]  # [batch, 25]

        # 全连接层输出分类结果
        x = self.fc2(x)  # [batch, 2]

        return x

class MouseNeuralNetwork2(nn.Module):
    def __init__(self, length_single_mouse_traj):
        super(MouseNeuralNetwork2, self).__init__()


        # 使用 ResNet1D 作为特征提取器
        self.resnet = resnet50_1d(feature_extractor=False)  # 特征提取模式


    def forward(self, input):
        # input shape: (batch_size, num_features=11, sequence_length)
        # 通过 ResNet1D 提取特征
        x = self.resnet(input)  # [batch, 512]

        return x
