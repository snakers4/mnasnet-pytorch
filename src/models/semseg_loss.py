import torch
import torch.nn as nn
import torch.nn.functional as F

class HardDice(nn.Module):
    def __init__(self,
                 threshold=0.5):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, outputs, targets):
        eps = 1e-10
        
        dice_target = (targets == 1).float()
        dice_output = F.sigmoid(outputs)    
        hard_output = (dice_output > self.threshold).float()
        
        intersection = (hard_output * dice_target).sum()
        union = hard_output.sum() + dice_target.sum() + eps
        hard_dice = (1+torch.log(2 * intersection / union))
        hard_dice = torch.clamp(hard_dice,0,1)
        
        return hard_dice

class SemsegLoss(nn.Module):
    def __init__(self,
                 use_running_mean=False,
                 bce_weight=1,
                 dice_weight=1,
                 eps=1e-10,
                 gamma=0.9
                 ):
        super().__init__()

        self.nll_loss = nn.BCEWithLogitsLoss()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.gamma = gamma 
        
        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        if self.use_running_mean == True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def reset_parameters(self):
        self.running_bce_loss.zero_()        
        self.running_dice_loss.zero_()            

    def forward(self, outputs, targets):
        # inputs and targets are assumed to be BxCxWxH
        assert len(outputs.shape) == len(targets.shape)
        # assert that B, W and H are the same
        assert outputs.size(-0) == targets.size(-0)
        assert outputs.size(-1) == targets.size(-1)
        assert outputs.size(-2) == targets.size(-2) 
        
        bce_loss = self.nll_loss(outputs, targets)

        dice_target = (targets == 1).float()
        dice_output = F.sigmoid(outputs)
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + self.eps
        dice_loss = (-torch.log(2 * intersection / union))        
        
        if self.use_running_mean == False:
            bmw = self.bce_weight
            dmw = self.dice_weight
            # loss += torch.clamp(1 - torch.log(2 * intersection / union),0,100)  * self.dice_weight
        else:
            self.running_bce_loss = self.running_bce_loss * self.gamma + bce_loss.data * (1 - self.gamma)        
            self.running_dice_loss = self.running_dice_loss * self.gamma + dice_loss.data * (1 - self.gamma)

            bm = float(self.running_bce_loss)
            dm = float(self.running_dice_loss)

            bmw = 1 - bm / (bm + dm)
            dmw = 1 - dm / (bm + dm)
                
        loss = bce_loss * bmw + dice_loss * dmw
        
        return loss,bce_loss,dice_loss

class SemsegLossWeighted(nn.Module):
    def __init__(self,
                 use_running_mean=False,
                 bce_weight=1,
                 dice_weight=1,
                 eps=1e-10,
                 gamma=0.9,
                 use_weight_mask=False,                 
                 ):
        super().__init__()

        self.use_weight_mask = use_weight_mask
        
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.gamma = gamma 
        
        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        if self.use_running_mean == True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def reset_parameters(self):
        self.running_bce_loss.zero_()        
        self.running_dice_loss.zero_()            

    def forward(self,
                outputs,
                targets,
                weights):
        # inputs and targets are assumed to be BxCxWxH
        assert len(outputs.shape) == len(targets.shape)
        # assert that B, W and H are the same
        assert outputs.size(0) == targets.size(0)
        assert outputs.size(2) == targets.size(2)
        assert outputs.size(3) == targets.size(3)
        
        # weights are assumed to be BxWxH
        # assert that B, W and H are the are the same for target and mask
        assert outputs.size(0) == weights.size(0)
        assert outputs.size(2) == weights.size(1)
        assert outputs.size(3) == weights.size(2)
        
        if self.use_weight_mask:
            bce_loss = F.binary_cross_entropy_with_logits(input=outputs,
                                                          target=targets,
                                                          weight=weights)            
        else:
            bce_loss = self.nll_loss(input=outputs,
                                     target=targets)

        dice_target = (targets == 1).float()
        dice_output = F.sigmoid(outputs)
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + self.eps
        dice_loss = (-torch.log(2 * intersection / union))        
        
        if self.use_running_mean == False:
            bmw = self.bce_weight
            dmw = self.dice_weight
            # loss += torch.clamp(1 - torch.log(2 * intersection / union),0,100)  * self.dice_weight
        else:
            self.running_bce_loss = self.running_bce_loss * self.gamma + bce_loss.data * (1 - self.gamma)        
            self.running_dice_loss = self.running_dice_loss * self.gamma + dice_loss.data * (1 - self.gamma)

            bm = float(self.running_bce_loss)
            dm = float(self.running_dice_loss)

            bmw = 1 - bm / (bm + dm)
            dmw = 1 - dm / (bm + dm)
                
        loss = bce_loss * bmw + dice_loss * dmw
        
        return loss,bce_loss,dice_loss    