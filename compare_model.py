import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DClassifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(CNN1DClassifier, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 第三层卷积
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # 第四层卷积
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        
        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # 第一层
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # 第二层
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # 第三层
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # 第四层
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # 展平
        x = torch.mean(x, dim=2)  # 全局平均池化
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        
        # 定义LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # LSTM前向传播
        # x的形状: (batch_size, sequence_length, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        
        # 全连接层进行分类
        x = F.relu(self.fc1(last_out))
        x = self.fc2(x)
        
        return x

