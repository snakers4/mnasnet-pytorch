import math
import torch
import torch.nn as nn
from models.mnasnet import Mnasnet
import torchvision.models as models

def load_model(arch='resnet18',
               pretrained=True):
    if arch.startswith('resnet'):
        model = models.__dict__[arch](pretrained=pretrained)
        print('Resnet initialized')        
    elif arch.startswith('mnasnet'):
        model = Mnasnet(cut_channels_first=False)
        print('Mnasnet initialized')
    else:
        raise("Finetuning not supported on this architecture yet") 
    return model
  
class FineTuneModelPool(nn.Module):
    def __init__(self,
                 original_model,
                 arch,
                 num_classes,
                 classifier_config):
        super(FineTuneModelPool, self).__init__()

        self.num_classes = num_classes
        
        if arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(
                *list(original_model.children())[:-3],
            )
            if arch == 'resnet152':
                self.modelName = 'resnet152'
                final_feature_map = 1024
            elif arch == 'resnet50':
                self.modelName = 'resnet50'
                final_feature_map = 1024
            else:
                raise ValueError("Finetuning not supported on this architecture yet")
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1)
            )
        elif arch.startswith('mnasnet'):
            # Everything except the last linear layer
            self.features = original_model.features
            final_feature_map = 320
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1)
            )
            self.modelName = 'mnasnet'
        else:
            raise ValueError("Finetuning not supported on this architecture yet")
            
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
        elif classifier_config == '320':
            self.classifier = nn.Sequential(
                nn.Linear(final_feature_map, num_classes)
            )
        else:
            raise ValueError("Finetuning not supported on this architecture yet")            
            
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)            

    
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
        f = self.pooling(f)
        y = self.classifier(f.view(f.size(0), -1)) 
        return y    
    
