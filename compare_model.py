import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DClassifier(nn.Module):
    def __init__(self):
        super(CNN1DClassifier, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        # 第二层卷积层
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        # 第三层卷积层
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # 第四层卷积层
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 100, 256)
        self.fc2 = nn.Linear(256, 2)  # 二分类输出
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 输入 x: (batch_size, 1, 100)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 16, 100)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 32, 100)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch_size, 64, 100)
        x = F.relu(self.bn4(self.conv4(x)))  # (batch_size, 128, 100)
        
        # 展平
        x = x.view(x.size(0), -1)  # (batch_size, 12800)
        
        x = F.relu(self.fc1(x))  # (batch_size, 256)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch_size, 2)
        
        return x
