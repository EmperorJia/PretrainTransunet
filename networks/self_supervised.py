import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random
import os
import numpy as np

class myUNet(nn.Module):

    def __init__(self, num_class, multi_task=False, num_multi_task_class=1):
        super().__init__()
        num_channel = [8, 16, 32, 64, 128]
        self.num_class = num_class
        self.multi_task = multi_task
        self.num_multi_task_class = num_multi_task_class

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = ResTwoLayerConvBlock(1, num_channel[0], num_channel[0])
        self.conv1_0 = ResTwoLayerConvBlock(
            num_channel[0], num_channel[1], num_channel[1])
        self.conv2_0 = ResTwoLayerConvBlock(
            num_channel[1], num_channel[2], num_channel[2])
        self.conv3_0 = ResTwoLayerConvBlock(
            num_channel[2], num_channel[3], num_channel[3])
        self.conv4_0 = ResTwoLayerConvBlock(
            num_channel[3], num_channel[4], num_channel[4])

        self.conv3_1 = ResTwoLayerConvBlock(
            num_channel[3] + num_channel[4], num_channel[3], num_channel[3])
        self.conv2_2 = ResTwoLayerConvBlock(
            num_channel[2] + num_channel[3], num_channel[2], num_channel[2])
        self.conv1_3 = ResTwoLayerConvBlock(
            num_channel[1] + num_channel[2], num_channel[1], num_channel[1])
        self.conv0_4 = ResTwoLayerConvBlock(
            num_channel[0] + num_channel[1], num_channel[0], num_channel[0])

        self.final = nn.Conv3d(
            num_channel[0], num_class, kernel_size=1, bias=False)
        self.final1 = nn.Conv3d(
            num_channel[2], num_multi_task_class, kernel_size=(8, 40, 40))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x))
        x2_0 = self.conv2_0(self.pool(x1_0))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_0 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_0 = self.conv2_2(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_0 = self.conv1_3(torch.cat([x1_0, self.up(x2_0)], 1))

        x = self.conv0_4(torch.cat([x, self.up(x1_0)], 1))
        x = self.final(x)

        return x

class ResTwoLayerConvBlock(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2):
        super(ResTwoLayerConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, inter_channel,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(inter_channel)
        self.conv2 = nn.Conv3d(inter_channel, out_channel,
                               kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channel)
        self.conv3 = nn.Conv3d(in_channel, out_channel,
                               kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p=p, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.drop(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        x = self.conv3(x)
        x = self.norm2(x)

        out += x
        out = self.relu(out)
        return out

class ResTwoLayerConvBlock2d(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2):
        super(ResTwoLayerConvBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, inter_channel,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(inter_channel)
        self.conv2 = nn.Conv2d(inter_channel, out_channel,
                               kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=p, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.drop(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        x = self.conv3(x)
        x = self.norm2(x)

        out += x
        out = self.relu(out)
        return out

class myUNet2d(nn.Module):

    def __init__(self, num_class, multi_task=False, num_multi_task_class=1):
        super().__init__()
        num_channel = [8, 16, 32, 64, 128]
        self.num_class = num_class
        self.multi_task = multi_task
        self.num_multi_task_class = num_multi_task_class

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ResTwoLayerConvBlock2d(1, num_channel[0], num_channel[0])
        self.conv1_0 = ResTwoLayerConvBlock2d(num_channel[0], num_channel[1], num_channel[1])
        self.conv2_0 = ResTwoLayerConvBlock2d(num_channel[1], num_channel[2], num_channel[2])
        self.conv3_0 = ResTwoLayerConvBlock2d(num_channel[2], num_channel[3], num_channel[3])
        self.conv4_0 = ResTwoLayerConvBlock2d(num_channel[3], num_channel[4], num_channel[4])

        self.conv3_1 = ResTwoLayerConvBlock2d(num_channel[3] + num_channel[4], num_channel[3], num_channel[3])
        self.conv2_2 = ResTwoLayerConvBlock2d(num_channel[2] + num_channel[3], num_channel[2], num_channel[2])
        self.conv1_3 = ResTwoLayerConvBlock2d(num_channel[1] + num_channel[2], num_channel[1], num_channel[1])
        self.conv0_4 = ResTwoLayerConvBlock2d(num_channel[0] + num_channel[1], num_channel[0], num_channel[0])

        self.final = nn.Conv2d(num_channel[0], num_class, kernel_size=1, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x))
        x2_0 = self.conv2_0(self.pool(x1_0))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_0 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_0 = self.conv2_2(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_0 = self.conv1_3(torch.cat([x1_0, self.up(x2_0)], 1))

        x = self.conv0_4(torch.cat([x, self.up(x1_0)], 1))
        x = self.final(x)

        return x


'''丐版resnet'''
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)



    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 3)



    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

'''ResNet-50'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


'''VIT_2D'''
class VIT2D(nn.Module):
    def __init__(self, input_dim, num_classes, patch_size, hidden_dim, num_heads, num_layers):
        super(VIT2D, self).__init__()
        self.embedding = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, hidden_dim, int(160/patch_size[0]), int(160/patch_size[1])))
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Input x has shape (B, C, H, W)
        x = self.embedding(x)  # (B, hidden_dim, D//patch_size, H//patch_size, W//patch_size)

        x = x + self.position_embedding
        # Reshape to (D*H*W, B, hidden_dim)
        x = x.permute(2, 3, 0, 1).contiguous().view(x.size(2) * x.size(3), x.size(0), x.size(1))

        # Transformer
        x = self.transformer(x, x)  # 输入和目标都是x

        # Reshape back to (B, D*H*W, hidden_dim)
        x = x.view(x.size(1), x.size(0), x.size(2))

        # Average pooling along the spatial dimensions
        x = torch.mean(x, dim=1)

        # Fully connected layer
        x = self.fc(x)

        return x

'''ResNet-50-3d'''
class ResidualBlock_3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock_3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class ResNet50_3d(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50_3d, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock_3d(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock_3d(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

'''VIT_3D'''
class VIT3D(nn.Module):
    def __init__(self, input_dim, num_classes, patch_size, hidden_dim, num_heads, num_layers):
        super(VIT3D, self).__init__()
        self.embedding = nn.Conv3d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=patch_size,stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, hidden_dim, int(32/patch_size[0]), int(160/patch_size[1]), int(160/patch_size[2])))
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Input x has shape (B, C, D, H, W)
        x = self.embedding(x)  # (B, hidden_dim, D//patch_size, H//patch_size, W//patch_size)

        x = x + self.position_embedding
        # Reshape to (D*H*W, B, hidden_dim)
        x = x.permute(2, 3, 4, 0, 1).contiguous().view(x.size(2) * x.size(3) * x.size(4), x.size(0), x.size(1))

        # Transformer
        x = self.transformer(x, x)  # 输入和目标都是x

        # Reshape back to (B, D*H*W, hidden_dim)
        x = x.view(x.size(1), x.size(0), x.size(2))

        # Average pooling along the spatial dimensions
        x = torch.mean(x, dim=1)

        # Fully connected layer
        x = self.fc(x)

        return x


class SelfSupervisedCNN(nn.Module):
    def __init__(self):
        super(SelfSupervisedCNN, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # 定义解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        outputs = self.decoder(encoded)
        return outputs

    def get_encoder(self):
        return self.encoder

class Classifier_res(nn.Module):
    def __init__(self, num_classes=3):
        super(Classifier_res, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock_3d(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock_3d(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x