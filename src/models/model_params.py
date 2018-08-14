# LinkNet based models
from models.linknet import LinkNet18,LinkNet34,LinkNet50,LinkNet101,LinkNet152,LinkNeXt,LinkDenseNet121,LinkDenseNet161,CoarseLinkNet50
# Inception based models
from models.linknet import LinkInceptionResNet,LinkCeption
# Unet-based models
from models.unet import UnetResnet18,UnetResnet34,UNet11,UNet16,UnetResnet152,UnetResnet101

model_presets = {
    # inception-based models
    'linkception4' : [LinkCeption,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],
    'linkceptionresnet' : [LinkInceptionResNet,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],  
    
    # unet-based models
    'unet11' : [UNet11,{'num_classes':1,'pretrained':True}],
    'unet16' : [UNet16,{'num_classes':1,'is_deconv':False,'pretrained':True}],
    'uresnet18' : [UnetResnet18,{'num_classes':1,'is_deconv':False,'pretrained':True}],    
    'uresnet34' : [UnetResnet34,{'num_classes':1,'is_deconv':False,'pretrained':True}],
    'uresnet101' : [UnetResnet101,{'num_classes':1,'is_deconv':False,'pretrained':True, "num_filters":64}],    
    'uresnet152' : [UnetResnet152,{'num_classes':1,'is_deconv':False,'pretrained':True, "num_filters":64}],
    
    # linknet-based models
    'linknet18' : [LinkNet18,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],    
    'linknet34' : [LinkNet34,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],
    'linknet50' : [LinkNet50,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],
    'coarse_linknet50' : [CoarseLinkNet50,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],    
    
    # heavier linknet-based models
    'linknext' : [LinkNeXt,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],
    'linkdensenet' : [LinkDenseNet121,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],
    'linkdensenet161' : [LinkDenseNet161,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],     
    'linknet101' : [LinkNet101,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],    
    'linknet152' : [LinkNet152,{'num_channels':3,'num_classes':1,'is_deconv':False,'pretrained':True,'decoder_kernel_size':4}],
}
