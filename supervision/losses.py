import torch
import torch.nn as nn
import torch.nn.functional as F

def charbonnier_penalty(x, epsilon_squared=0.01):
    charbonnier_loss = torch.sqrt(x * x + epsilon_squared)
    return charbonnier_loss

def KL_divergence(logvar, mu):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()

def dice_loss(pred, target):
    smooth = 0.1

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    loss = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean()
    
    return 1 - loss


def dice_score(pred, target):
    smooth = 0.1

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    score = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean()
    
    return score


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1).long()

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()