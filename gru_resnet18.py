import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_18 import BasicBlock1D, ResNet1D

class ResNetGRU(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
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

class ResNetGRU(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
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


# # Example Usage
# model = ResNetGRU(num_classes=5)
# inputs = torch.randn(32, 1, 100)  # Batch size: 32, Channels: 1, Sequence length: 100
# outputs = model(inputs)
# print(outputs.shape)  # Expected output: (32, 5)


# # Example Usage
# model = ResNetGRU(num_classes=5)
# inputs = torch.randn(32, 1, 100)  # Batch size: 32, Channels: 1, Sequence length: 100
# outputs = model(inputs)
# print(outputs.shape)  # Expected output: (32, 5)
