# resnet_mouse.py

import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=2, feature_extractor=True):
        """
        Args:
            block: Block class (e.g., BasicBlock1D)
            layers: List containing number of blocks in each layer
            num_classes: Number of output classes
            feature_extractor: If True, the model acts as a feature extractor (no final fc layer)
        """
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        # 修改输入通道数为 11
        self.conv1 = nn.Conv1d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_extractor = feature_extractor
        if not self.feature_extractor:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, 11, sequence_length)
        x = self.conv1(x)    # [batch, 64, L/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [batch, 64, L/4]

        x = self.layer1(x)   # [batch, 64, L/4]
        x = self.layer2(x)   # [batch, 128, L/8]
        x = self.layer3(x)   # [batch, 256, L/16]
        x = self.layer4(x)   # [batch, 512, L/32]

        x = self.avgpool(x)  # [batch, 512, 1]
        x = torch.flatten(x, 1)  # [batch, 512]

        if self.feature_extractor:
            return x  # 返回特征向量
        else:
            x = self.fc(x)  # [batch, num_classes]
            return x

def resnet50_1d(feature_extractor=True):
    """
    创建 ResNet1D-50 模型

    Args:
        feature_extractor: 如果为 True，则模型作为特征提取器使用（不包含最终的全连接层）

    Returns:
        ResNet1D 模型实例
    """
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], num_classes=2, feature_extractor=feature_extractor)
