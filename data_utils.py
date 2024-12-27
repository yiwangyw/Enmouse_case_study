# data_utils.py
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from config import Config

def ensure_cpu_tensor(tensor):
    """确保张量在CPU上"""
    return tensor.cpu() if tensor.is_cuda else tensor

class GetLoader(Dataset):
    """自定义数据集加载器"""
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
        
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        # 确保返回的张量在CPU上
        if isinstance(data, torch.Tensor) and data.is_cuda:
            data = data.cpu()
        if isinstance(labels, torch.Tensor) and labels.is_cuda:
            labels = labels.cpu()
        return data, labels
    
    def __len__(self):
        return len(self.data)

def read_test_data_shape(test_dataloader):
    """读取测试数据的形状"""
    try:
        # 获取第一个批次的数据
        data_iter = iter(test_dataloader)
        inputs, _ = next(data_iter)
        return inputs.shape
    except StopIteration:
        raise RuntimeError("Empty dataloader")
    except Exception as e:
        print(f"Error reading test data shape: {str(e)}")
        raise

def insert_new_test_data(test_dataloader, X_insert, label_insert, batch_size):
    """插入新的测试数据"""
    try:
        # 确保数据在CPU上
        X_insert = ensure_cpu_tensor(X_insert)
        label_insert = ensure_cpu_tensor(label_insert)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X_insert, label_insert)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
    except Exception as e:
        print(f"Error inserting new test data: {str(e)}")
        raise