import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'FocalDiceLoss']

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, dice_weight=0.5, smooth=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.smooth = smooth
        
    def focal_loss(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()
    
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice
    
    def forward(self, pred, target):

        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total_loss = focal + self.dice_weight * dice
        
        return total_loss










