import torch
import torch.nn as nn
import torchvision.models as models
from models.InceptionResnet import inceptionresnetv2
import math

def load_model(arch='resnet18',
               pretrained=True):
    if arch.startswith('resnet') :
        model = models.__dict__[arch](pretrained=pretrained)
    else :
        raise("Finetuning not supported on this architecture yet") 
    return model

class ConvRelu(nn.Module):
    def __init__(self, in_, out, kernel_size, activation = nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, kernel_size=kernel_size)
        self.activation = activation
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
   
class FineTuneModelPool(nn.Module):
    def __init__(self,
                 original_model,
                 arch,
                 num_classes,
                 scale, 
                 is_mixed,
                 classifier_config):
        super(FineTuneModelPool, self).__init__()

        self.num_classes = num_classes
        self.is_mixed = is_mixed
        
        if arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(
                *list(original_model.children())[:-3],
            )
            if arch == 'resnet152':
                final_feature_map = 1024
            elif arch == 'resnet50':
                final_feature_map = 1024
            
            if self.is_mixed == True:
                
                if scale == 0.25:
                    # pool by the smallest possible
                    pooling_filter_size = 8
                    
                    self.pooling = nn.Sequential(
                        nn.AvgPool2d(pooling_filter_size,stride=1)
                    )                    
                    self.classifier_cl0 = nn.Sequential(
                        ConvRelu(final_feature_map,final_feature_map,(1,16-pooling_filter_size+1)),
                        ConvRelu(final_feature_map,final_feature_map,(8-pooling_filter_size+1,1))
                    )
                    self.classifier_cl1 = nn.Sequential(
                        ConvRelu(final_feature_map,final_feature_map,(1,8-pooling_filter_size+1)),
                        ConvRelu(final_feature_map,final_feature_map,(16-pooling_filter_size+1,1))
                    )            
                    self.classifier_cl2 = nn.Sequential(
                        ConvRelu(final_feature_map,final_feature_map,(1,12-pooling_filter_size+1)),
                        ConvRelu(final_feature_map,final_feature_map,(12-pooling_filter_size+1,1))
                    )    
                    
                elif scale == 0.5:
                    # pool by the smallest possible
                    pooling_filter_size = 16
                    
                    self.pooling = nn.Sequential(
                        nn.AvgPool2d(pooling_filter_size,stride=1)
                    )                    
                    self.classifier_cl0 = nn.Sequential(
                        ConvRelu(final_feature_map,final_feature_map,(1,32-pooling_filter_size+1)),
                        ConvRelu(final_feature_map,final_feature_map,(16-pooling_filter_size+1,1))
                    )
                    self.classifier_cl1 = nn.Sequential(
                        ConvRelu(final_feature_map,final_feature_map,(1,16-pooling_filter_size+1)),
                        ConvRelu(final_feature_map,final_feature_map,(32-pooling_filter_size+1,1))
                    )            
                    self.classifier_cl2 = nn.Sequential(
                        ConvRelu(final_feature_map,final_feature_map,(1,24-pooling_filter_size+1)),
                        ConvRelu(final_feature_map,final_feature_map,(24-pooling_filter_size+1,1))
                    )               
            else:
                self.pooling = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1)
                )
            
            if classifier_config == '256':
                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(final_feature_map, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(256, num_classes)
                )
            elif classifier_config == '512_256':
                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(final_feature_map, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),                    
                    nn.Linear(256, num_classes)
                )
            else:
                raise ValueError("Finetuning not supported on this architecture yet")
            self.modelName = 'resnet'
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)            
        else:
            raise ValueError("Finetuning not supported on this architecture yet")
    
    def freeze(self):
        print('Features frozen')
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze(self):
        print('Features unfrozen')
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = True            
            
    def forward(self, x):
        f = self.features(x)
        # print(f.shape)
        f = self.pooling(f)
        # print(f.shape)        
        if self.is_mixed == True:
            if x.shape[2:4] == (512, 1024) or x.shape[2:4] == (256, 512)  or x.shape[2:4] == (128, 256):
                f = self.classifier_cl0(f)
            elif x.shape[2:4] == (1024, 512) or x.shape[2:4] == (512, 256) or x.shape[2:4] == (256, 128):
                f = self.classifier_cl1(f)
            elif x.shape[2:4] == (768, 768) or x.shape[2:4] == (384, 384) or x.shape[2:4] == (192, 192):
                f = self.classifier_cl2(f)
            else:
                raise ValueError("Wrong image format")
        # print(f.shape)
        y = self.classifier(f.view(f.size(0), -1)) 
        # print(y.shape)
        return y    
    
