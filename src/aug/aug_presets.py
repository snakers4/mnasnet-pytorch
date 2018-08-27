import imgaug as ia
from aug.cv2_augs import *  
from imgaug import augmenters as iaa    

class IaaAugs:
    def __init__(self, prob=0.5):
        self.prob = prob
        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = self.sometimes(iaa.Sequential([
                iaa.SomeOf((0, 2),
                    [
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)),
                            iaa.AverageBlur(k=(2, 7)),
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                        ),
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5),
                            iaa.CoarseDropout(
                                (0.03, 0.15), size_percent=(0.02, 0.05),
                                per_channel=0.2
                            ),
                        ]),
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                    ],
                    random_order=True
                )
            ],
            random_order=True
        ))        

    def __call__(self, img):
        if random.random() < self.prob:
            img = self.seq.augment_image(img)
        return img                              
                                
class TrainAugsIaa(object):
    def __init__(self,
                 prob=0.5,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.prob = prob
        self.mean = mean
        self.std = std
    def __call__(self, img, mask, target_resl):
        
        aug_list = []
        if target_resl is not None:
            aug_list.append(Resize(size=target_resl))
            
        aug_list.extend([
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, prob=self.prob),
            RandomFlip(prob=0.5),
            ImageOnly(RandomContrast(limit=0.2, prob=self.prob)),
            ImageOnly(RandomFilter(limit=0.5, prob=self.prob/2)),
            ImageOnly(IaaAugs(prob=self.prob/2)),            
            ImageOnly(Normalize(mean=self.mean, std=self.std)),
        ])  
        
        return DualCompose(aug_list)(img, mask)

class TrainAugs(object):
    def __init__(self,
                 prob=0.5,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.prob = prob
        self.mean = mean
        self.std = std
    def __call__(self,
                 img,
                 mask,
                 target_img_resl,
                 target_msk_resl):
        
        aug_list = []
        
        if target_img_resl is not None:
            aug_list.append(ImageOnly(Resize(size=target_img_resl)))
            
        if target_msk_resl is not None:
            aug_list.append(MaskOnly(Resize(size=target_msk_resl)))
        
        aug_list.extend([
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=0, prob=self.prob),
            RandomFlip(prob=self.prob),
            ImageOnly(RandomContrast(limit=0.2, prob=self.prob)),
            ImageOnly(RandomFilter(limit=0.5, prob=self.prob/2)),
            ImageOnly(Normalize(mean=self.mean, std=self.std)),
        ])          
        
        return DualCompose(aug_list)(img, mask)     
    
class ValAugs(object):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    def __call__(self,
                 img,
                 mask,
                 target_img_resl,
                 target_msk_resl):
        
        aug_list = []
        
        if target_img_resl is not None:
            aug_list.append(ImageOnly(Resize(size=target_img_resl)))
            
        if target_msk_resl is not None:
            aug_list.append(MaskOnly(Resize(size=target_msk_resl)))     

        aug_list.extend([
            ImageOnly(Normalize(mean=self.mean, std=self.std)),
        ])             

        return DualCompose(aug_list)(img, mask)