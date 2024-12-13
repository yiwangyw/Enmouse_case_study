import sys
import json
import torch 
from torch.utils import data
from torch.autograd import Variable
import torchvision
from torchvision.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import csv
from torch.utils.data import random_split
from torch.utils.data import Dataset

sys.path.append("/root/Mouse/model")  # 换成你自己的绝对路径
from resnet_mouse import _ResNetLayer
from mouse_traj_classification import MouseNeuralNetwork, MouseNeuralNetwork2

class MouseTrajectoryDataset(Dataset):
    def __init__(self, json_file, is_positive=True):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.sequence_length = data['metadata']['sequence_length']
        self.feature_names = data['metadata']['feature_names']
        
        trajectories = []
        for sample in data['samples']:
            trajectory = np.array([[step[feature] for feature in self.feature_names] 
                                 for step in sample])
            trajectories.append(trajectory)
        
        self.data = np.array(trajectories)
        self.data = torch.FloatTensor(self.data).transpose(1, 2)
        self.labels = torch.zeros(len(self.data)) if is_positive else torch.ones(len(self.data))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def load_mouse_data(json_file):
    """
    加载鼠标轨迹数据并进行预处理
    Args:
        json_file: JSON文件路径
    Returns:
        torch.Tensor: 形状为(num_samples, num_features, sequence_length)的张量
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取数据维度信息
    sequence_length = data['metadata']['sequence_length']
    feature_names = data['metadata']['feature_names']
    
    # 转换数据格式
    trajectories = []
    for sample in data['samples']:
        trajectory = np.array([[step[feature] for feature in feature_names] 
                             for step in sample])
        trajectories.append(trajectory)
    
    # 转换为 (num_samples, sequence_length, num_features) 的数组
    data = np.array(trajectories)
    # 转换为 (num_samples, num_features, sequence_length) 的张量
    return torch.FloatTensor(data).transpose(1, 2)

def process_mouse_data(positive_file, negative_file):
    """
    处理正负样本数据
    Args:
        positive_file: 正样本JSON文件路径
        negative_file: 负样本JSON文件路径
    Returns:
        tuple: (X, label) 处理后的数据和标签
    """
    # 加载数据
    positive_data = load_mouse_data(positive_file)
    negative_data = load_mouse_data(negative_file)
    
    # 创建标签
    labels_positive = torch.zeros(len(positive_data))
    labels_negative = torch.ones(len(negative_data))
    
    # 合并数据
    X = torch.cat([positive_data, negative_data], dim=0)
    label = torch.cat([labels_positive, labels_negative], dim=0)
    
    return X, label


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

def insert_new_test_data(test_dataloader,X_insert,label_insert,target_batch_size):
    '''
    Args:
    test_dataloader: torch.utils.data.DataLoader original test dataloader
    X_insert: new test X
    label_insert: new test label
    target_batch_size: int

    Return:
    X_test_loader: torch.utils.data.DataLoader new test dataloader

    '''
    X_new = X_insert
    label_new = label_insert
    for data in test_dataloader:
        X_origin, label_origin = data
        X_new = torch.cat([X_origin, X_new], dim=0)
        label_new = torch.cat([label_origin, label_new], dim=0).to(torch.int64)
    test_dataset_new = torch.utils.data.TensorDataset(X_new, label_new)
    # 修改这里，移除pin_memory_device参数
    X_test_loader = torch.utils.data.DataLoader(test_dataset_new, 
                                              batch_size=target_batch_size,
                                              shuffle=True)
    return X_test_loader

def read_test_data_shape(test_dataloader):
    '''
    args:
    test_dataloader: torch.utils.data.DataLoader original test dataloader

    return:
    shape of data [batch_size, data_shape...]
    '''
    for data in test_dataloader:
        X, label = data
        break
    return X.shape


