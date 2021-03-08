
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MaxPLoss(nn.Module):

    def __init__(self, ord_num, gamma, beta, discretization="SID"):
        super(MaxPLoss, self).__init__()
        self.ord_num = ord_num
        self.gamma = gamma
        self.beta = beta
        self.discretization = discretization

        level_vector = torch.arange(self.ord_num+1).float() # if not using .float(), the depth_vector are integers
        if self.discretization == "SID":
            depth_vector = math.log(self.beta) * level_vector / self.ord_num
            # depth_vector = torch.exp(np.log(self.beta) * level_vector / self.ord_num)
        else:
            depth_vector = 1.0 + (self.beta - 1.0) * level_vector / self.ord_num

        depth_vector = (depth_vector[:-1] + depth_vector[1:]) / 2
        self.register_buffer("depth_vector", depth_vector)

    def __call__(self, p_bin, gt):
        """
        p_bin: N*[ord_num]*H*W, each channel in [] represents the probability that the depth fall in that range (processed by softmax)
        """
        valid_mask = (gt > 0).detach()
        gt_gamma = gt + self.gamma
        if self.discretization == "SID":
            gt_gamma = torch.log(torch.clamp(gt_gamma, min=1e-5))

        depth_pred = (p_bin * self.depth_vector.view(1, -1, 1, 1)).sum(1) # N*H*W

        # if self.discretization == "SID":
        #     depth_pred = torch.exp(depth_pred) 
        # depth_pred = depth_pred - self.gamma

        diff = gt_gamma - depth_pred
        # diff_masked = diff[valid_mask]
        loss_mean = torch.sqrt(diff[valid_mask].pow(2).sum() / valid_mask.sum())

        depth_dev_square = (self.depth_vector.view(1, -1, 1, 1) - gt_gamma.unsqueeze(1))**2

        depth_deviation = torch.sqrt((p_bin * depth_dev_square ).sum(1)) # N*H*W
        loss_deviation = depth_deviation[valid_mask].sum() / valid_mask.sum()

        sigma = torch.sqrt(depth_deviation).unsqueeze(1).detach()    # N*1*H*W
        likelihood = torch.exp(-0.5 * depth_dev_square / (sigma**2) ) / sigma / math.sqrt(2*math.pi)
        neg_log_like = - torch.log((p_bin * likelihood).sum(1)) # N*H*W
        loss_nll = neg_log_like[valid_mask].sum() / valid_mask.sum()

        losses = dict()
        losses['loss_mean'] = loss_mean
        losses['loss_deviation'] = loss_deviation
        losses['loss_nll'] = loss_nll

        return losses, depth_pred