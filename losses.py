import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F

def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)

    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
    
    def forward(self, sp: torch.Tensor, sn: torch.Tensor):
        print(f"sp: {sp.shape}, sn: {sn.shape}")
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)
        print(f"ap: {ap.shape}, an: {an.shape}")

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.gamma * ap * (sp - delta_p)
        logit_n = self.gamma * an * (sn - delta_n)
        print(f"logit_p: {logit_p.shape}, logit_n: {logit_n.shape}")

        loss = F.softplus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        print(f"loss: {loss}")
        return loss

class ContrastiveLoss(nn.Module):
    '''
    Takes embeddings of 2 samples and a target label == 1 if same class
    and label == 0 otherwise
    '''
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
    
    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances + 
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
    
class TripletLoss(nn.Module):
    '''
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    '''
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()