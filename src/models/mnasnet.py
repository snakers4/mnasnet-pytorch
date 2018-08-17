
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as data


# In[411]:


class SepConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, reduce=False, repeat=0):
        super(SepConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.repeat = repeat
        self.padding = self.kernel_size // 2
        self.stride = 2 if reduce else 1
        
        self.sequence = [nn.Conv2d(self.in_channels, self.in_channels, 
                                   self.kernel_size, self.stride, padding=self.padding, groups=self.in_channels), 
                         nn.Conv2d(self.in_channels, self.in_channels, 1, stride=1)] * self.repeat + \
                        [nn.Conv2d(self.in_channels, self.in_channels, 
                                   self.kernel_size, self.stride, padding=self.padding, groups=self.in_channels), 
                         nn.Conv2d(self.in_channels, self.out_channels, 1, stride=1)]
        self.sequence = nn.Sequential(*self.sequence)

    
    def forward(self, input):
        
        output = self.sequence(input)
        
        return output


# In[420]:


class MBConv_block(nn.Module):
    
    def __init__(self, in_channels, channel_factor, out_channels=None, kernel_size=3, reduce=False):
        super(MBConv_block, self).__init__()
    
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else self.in_channels
        self.kernel_size = kernel_size
        self.channel_factor = channel_factor
        self.padding = self.kernel_size // 2
        self.stride = 2 if reduce else 1
        
        self.sequence = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels*self.channel_factor, 1, stride=1),
                                      nn.Conv2d(self.in_channels*self.channel_factor, self.in_channels * self.channel_factor, \
                                                self.kernel_size, self.stride, padding=self.padding, groups=self.in_channels*self.channel_factor),
                                      nn.Conv2d(self.in_channels*self.channel_factor, self.out_channels, 1, stride=1))
                                      
    def forward(self, input):
        
        if self.in_channels == self.out_channels:
            output = input + self.sequence(input)
        else: output = self.sequence(input)
        
        return output      


# In[421]:


class MBConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, channel_factor, layers, kernel_size=3, reduce=True):
        super(MBConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.channel_factor = channel_factor
        self.reduce = reduce
        
        self.sequence = [MBConv_block(self.in_channels, self.channel_factor, self.out_channels, self.kernel_size, reduce=self.reduce)] +                         [MBConv_block(self.out_channels, self.channel_factor, kernel_size=self.kernel_size)] * (layers-1)
        self.sequence = nn.Sequential(*self.sequence)
        
    def forward(self, input):
        
        output = self.sequence(input)
        
        return output


# In[422]:


class Mnasnet(nn.Module):
    
    def __init__(self):
        super(Mnasnet, self).__init__()
        
        self.sequence = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                                      SepConv(32, 16, kernel_size=3),
                                      MBConv(16, 24, channel_factor=3, layers=3, kernel_size=3, reduce=True),
                                      MBConv(24, 40, channel_factor=3, layers=3, kernel_size=5, reduce=True),
                                      MBConv(40, 80, channel_factor=6, layers=3, kernel_size=5, reduce=True),
                                      MBConv(80, 96, channel_factor=6, layers=2, kernel_size=3, reduce=False),
                                      MBConv(96, 192, channel_factor=6, layers=4, kernel_size=5, reduce=True),
                                      MBConv(192, 320, channel_factor=6, layers=1, kernel_size=3, reduce=False)
                                     )
    
    def forward(self, input):
        
        output = self.sequence(input)
        return output

