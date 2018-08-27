import os
import sys
import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.model_zoo as model_zoo

default_activation = nn.ReLU6
debug_global = False

__all__ = ['mnasnet', 'MNasNet']

pretrained_settings = {
    'mnasnet': {
        'imagenet': {
            'url': '',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        },
        'openimages_multi': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 74
        }
    }
}

class ConvBlock(nn.Module):
    def __init__(self,
                 in_,
                 out_,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 activation=default_activation):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_,
                              out_,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_)
        self.activation = activation(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class SepConv(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_size=3,
                 reduce=False,
                 repeat=0):
        super(SepConv, self).__init__()
        
        padding = kernel_size // 2
        stride = 2 if reduce else 1
        
        self.sequence = [ConvBlock(in_=in_channels,
                                   out_=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channels),
                         ConvBlock(in_=in_channels,
                                   out_=in_channels,
                                   kernel_size=1,
                                   stride=1)] * repeat + \
                        [ConvBlock(in_=in_channels,
                                   out_=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channels),
                         ConvBlock(in_=in_channels,
                                   out_=out_channels,
                                   kernel_size=1,
                                   stride=1)]

        self.sequence = nn.Sequential(*self.sequence)
    
    def forward(self, input):
        output = self.sequence(input)
        if debug_global:
            print(output.shape)
        return output

class MBConv_block(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_factor,
                 out_channels=None,
                 kernel_size=3,
                 reduce=False):
        super(MBConv_block, self).__init__()
    
        self.in_channels = in_channels
        out_channels = out_channels if out_channels else in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2
        stride = 2 if reduce else 1
        
        self.sequence = nn.Sequential(ConvBlock(in_=in_channels,
                                                out_=in_channels*channel_factor,
                                                kernel_size=1,
                                                stride=1),
                                      ConvBlock(in_=in_channels*channel_factor,
                                                out_=in_channels * channel_factor,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                groups=in_channels*channel_factor),
                                      ConvBlock(in_=in_channels*channel_factor,
                                                out_=out_channels,
                                                kernel_size=1,
                                                stride=1))
                                      
    def forward(self, input):
        if self.in_channels == self.out_channels:
            output = input + self.sequence(input)
        else: 
            output = self.sequence(input)
        if debug_global:
            print(output.shape)            
        return output      

class MBConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 channel_factor,
                 layers,
                 kernel_size=3,
                 reduce=True):
        super(MBConv, self).__init__()
        
        
        self.sequence = [MBConv_block(in_channels,
                                      channel_factor,
                                      out_channels,
                                      kernel_size,
                                      reduce=reduce)] + \
                        [MBConv_block(out_channels,
                                      channel_factor,
                                      kernel_size=kernel_size)] * (layers-1)
        
        self.sequence = nn.Sequential(*self.sequence)
        
    def forward(self, input):
        output = self.sequence(input)
        return output

class Mnasnet(nn.Module):
    def __init__(self):
        super(Mnasnet, self).__init__()
        
        self.features = nn.Sequential(ConvBlock(3, 32, kernel_size=3, stride=2, padding=1),
                                      SepConv(32, 16, kernel_size=3),
                                      MBConv(16, 24, channel_factor=3, layers=3, kernel_size=3, reduce=True),
                                      MBConv(24, 40, channel_factor=3, layers=3, kernel_size=5, reduce=True),
                                      MBConv(40, 80, channel_factor=6, layers=3, kernel_size=5, reduce=True),
                                      MBConv(80, 96, channel_factor=6, layers=2, kernel_size=3, reduce=False),
                                      MBConv(96, 192, channel_factor=6, layers=4, kernel_size=5, reduce=True),
                                      MBConv(192, 320, channel_factor=6, layers=1, kernel_size=3, reduce=False)
                                     )
        
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.features(input)
        return output