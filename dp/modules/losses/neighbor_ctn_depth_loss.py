
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class NeighborCtnDepthLoss(nn.Module):

    def __init__(self, ord_num, gamma, beta, discretization="SID"):
        super(NeighborCtnDepthLoss, self).__init__()
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

    def __call__(self, p_bin, label, gt):
        """
        p_bin: N*[ord_num]*H*W, each channel in [] represents the probability that the depth fall in that range (processed by softmax)
        """
        valid_mask = (gt > 0).detach()
        gt_gamma = gt + self.gamma
        if self.discretization == "SID":
            gt_gamma = torch.log(torch.clamp(gt_gamma, min=1e-5))

        mask_zeros = torch.zeros_like(p_bin)
        mask_label = torch.arange(self.ord_num).view(1, -1, 1, 1).to(device=label.device)
        label = label.unsqueeze(1)
        p_bin_masked = torch.where( (mask_label >= label - 2) & (mask_label <= label + 2), p_bin, mask_zeros )
        p_bin_masked = F.normalize(p_bin_masked, dim=1, p=1)

        depth_pred = (p_bin_masked * self.depth_vector.view(1, -1, 1, 1)).sum(1) # N*H*W

        # if self.discretization == "SID":
        #     depth_pred = torch.exp(depth_pred) 
        # depth_pred = depth_pred - self.gamma

        diff = gt_gamma - depth_pred
        # diff_masked = diff[valid_mask]
        loss_mean = torch.sqrt(diff[valid_mask].pow(2).sum() / valid_mask.sum())

        losses = dict()
        losses['loss_mean'] = loss_mean
        losses['loss_deviation'] = torch.zeros_like(loss_mean)
        losses['loss_nll'] = torch.zeros_like(loss_mean)

        return losses, depth_pred