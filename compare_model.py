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


class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 2, 64)  # 双向LSTM，hidden_size翻倍
        self.fc2 = nn.Linear(64, 2)  # 二分类
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 输入 x: (batch_size, 1, 100)
        x = x.squeeze(1)  # (batch_size, 100)
        x = x.unsqueeze(2)  # (batch_size, 100, 1)
        
        # LSTM层
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, 100, 256)
        
        # 取最后一个时间步的输出（可以也使用平均池化等方式）
        final_output = lstm_out[:, -1, :]  # (batch_size, 256)
        
        # 全连接层
        x = F.relu(self.fc1(final_output))  # (batch_size, 64)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch_size, 2)
        
        return x

