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
        self.num_hidden_layer_gru = 25  #最开始是86、88、90、96、128
        self.sizeof_x = (length_single_mouse_traj - self.len_lost_in_mousetraj_single) * self.num_hidden_layer_gru

        self.conv0 = nn.Conv1d(1, 8, 5,padding=1)  # 50
        self.bn0 = nn.BatchNorm1d(8)
        ######################## task 2.2 ##########################
        self.layer1 = _ResNetLayer(in_channel=8, out_channel=16)#50-7    30-14=16  50-14=36
        self.layer2 = _ResNetLayer(in_channel=16, out_channel=32)#43-7    26-14
        self.layer3 = _ResNetLayer(in_channel=32, out_channel=64)#36-7=29   12-14
        self.layer4 = _ResNetLayer(in_channel=64, out_channel=128)#12-6
        ########################    END   ##########################
        self.fc = nn.Linear(40, 20)   #全连接
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.sizeof_x, 2)
        self.dr1 = nn.Dropout(0.4)
        self.dr2 = nn.Dropout(0.25)
        self.rnn = nn.GRU(
    input_size=16,
    hidden_size=self.num_hidden_layer_gru,
    num_layers=3,
    batch_first=False,
    bidirectional=False  # 改为True
)       
        self.sig = nn.Sigmoid()
        self.flatten = nn.Flatten(1)
    def forward(self, input):
        x = self.relu(self.bn0(self.conv0(input)))
        x = self.layer1(x)
        x = self.dr1(x)
        # x = self.layer2(x)
        # x = self.dr1(x)
        # x = self.layer3(x)
        # x = self.dr1(x)
        # x = self.layer4(x)
        # x = self.dr1(x)
        x = x.permute(0,2,1)
        x,self.hidden = self.rnn(x)
        x = x.reshape((x.size(0),-1))
        #print(x.shape)
        x = self.fc2(x)
        ########################    END   ##########################

        return x



class MouseNeuralNetwork2(nn.Module):
    def __init__(self, length_single_mouse_traj):
        super(MouseNeuralNetwork2, self).__init__()
        self.len_mouse_traj = length_single_mouse_traj
        self.conv0 = nn.Conv1d(1, 8, 5,padding=1)  # 50
        self.bn0 = nn.BatchNorm1d(8)
        self.end_channel_of_resnet3 = 64
        ######################## task 2.2 ##########################
        self.layer1 = _ResNetLayer(in_channel=8, out_channel=16)#50-14   80-14
        self.layer2 = _ResNetLayer(in_channel=16, out_channel=32)#36-14  66-14
        self.layer3 = _ResNetLayer(in_channel=32, out_channel=64)#22-14  52-14
     #   self.layer4 = _ResNetLayer(in_channel=64, out_channel=128)#12-6  38-14=24
        ########################    END   ##########################
        self.len_lost_in_mousetraj_all = 2    #14 28 42 #28 56 
        self.sizeof_x = (length_single_mouse_traj - self.len_lost_in_mousetraj_all) * self.end_channel_of_resnet3
       # print(self.sizeof_x)
        self.fc = nn.Linear(self.sizeof_x, 300)   #全连接 我试试300改成100
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(300, 2)
        self.dr1 = nn.Dropout(0.4)
        self.dr2 = nn.Dropout(0.25)
    def forward(self, input):
        x = self.relu(self.bn0(self.conv0(input)))
        x = self.layer1(x)
      #  print("x shape after layer1:", x.shape)
        x = self.dr1(x)
        x = self.layer2(x)
       # print("x shape after layer2:", x.shape)
        x = self.layer3(x)
      #  print("x shape after layer3:", x.shape)
      #  print(x.size())
        x = self.dr1(x)
       #x = self.layer4(x)
        x = x.view((x.size(0),-1)).squeeze()
      #  print(x.size())
        self.sizeof_x = x.size()
      #  print(self.sizeof_x.size())
        
        x = self.fc(x)
        x = self.fc2(x)
        # x = F.softmax(x, dim=1 )
        ########################    END   ##########################

        return x