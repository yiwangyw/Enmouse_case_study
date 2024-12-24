# mouse_traj_classification.py

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, WeightedRandomSampler, SequentialSampler, random_split
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

# 导入 ResNet1D
from resnet_mouse import BasicBlock1D, ResNet1D

class MouseNeuralNetwork(nn.Module):
    def __init__(self, length_single_mouse_traj, num_features=11):
        super(MouseNeuralNetwork, self).__init__()
        self.len_lost_in_mousetraj_single = 2
        self.num_hidden_layer_gru = 25

        # 使用 ResNet1D 作为特征提取器
        self.resnet = ResNetGRU()  # 特征提取模式

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
        self.resnet = ResNetGRU()  # 特征提取模式


    def forward(self, input):
        # input shape: (batch_size, num_features=11, sequence_length)
        # 通过 ResNet1D 提取特征
        x = self.resnet(input)  # [batch, 512]

        return x


class ResNetGRU(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super(ResNetGRU, self).__init__()
        self.resnet = ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=512, dropout_rate=dropout_rate)
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.fc = nn.Linear(256 * 2, num_classes)  # Bidirectional GRU

    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        x = self.resnet(x)  # Output shape: (batch_size, 512)
        x = x.unsqueeze(1)  # Add temporal dimension: (batch_size, 1, 512)
        x, _ = self.gru(x)  # Output shape: (batch_size, 1, 512)
        x = x[:, -1, :]  # Take the last GRU output
        x = self.fc(x)  # Output shape: (batch_size, num_classes)
        return x

class GRUResNet1D(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, dropout_rate=0.2):
        super(GRUResNet1D, self).__init__()
        
        # GRU for Temporal Feature Extraction
        self.gru = nn.GRU(input_size=5, hidden_size=hidden_size, num_layers=2, 
                          batch_first=True, dropout=dropout_rate, bidirectional=True)
        
        # ResNet for Spatial Feature Extraction
        self.resnet = ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=num_classes, dropout_rate=dropout_rate)
        
        # Final Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Input: (batch_size, 1, seq_len)
        x = x.squeeze(1)  # Remove the channel dimension -> (batch_size, seq_len)
        x = x.unsqueeze(-1)  # Add feature dimension -> (batch_size, seq_len, 1)
        print(x.shape)
        # GRU: Process sequential data
        x, _ = self.gru(x)  # Output: (batch_size, seq_len, hidden_size * 2)
        print(x.shape)
        
        # Reshape for ResNet (add channel dimension)
        x = x.permute(0, 2, 1)  # (batch_size, hidden_size * 2, seq_len)
        print(x.shape)
        x = self.resnet(x)  # Output: (batch_size, 512)
        
        # # Fully Connected Layer
        # x = self.dropout(x)
        # x = self.fc(x)  # Output: (batch_size, num_classes)
        
        return x

