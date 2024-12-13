# testdata_utils.py
import json
import torch
import numpy as np

def load_predict_data(json_file):
    """
    加载预测数据并进行预处理
    Args:
        json_file: JSON文件路径
    Returns:
        torch.Tensor: 形状为(num_samples, 1, num_features, sequence_length)的张量
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
    # 转换为 (num_samples, 1, num_features, sequence_length) 的张量
    data = torch.FloatTensor(data).transpose(1, 2)
    data = data.unsqueeze(1)  # 添加channel维度
    return data

def process_predict_data(predict_file):
    """
    处理预测样本数据
    Args:
        predict_file: 预测样本JSON文件路径
    Returns:
        tuple: (X, label) 处理后的数据和标签
    """
    # 加载数据
    X = load_predict_data(predict_file)
    
    # 创建标签(全为0)
    labels = torch.zeros(len(X))
    
    # 标准化数据
    
    return X, labels