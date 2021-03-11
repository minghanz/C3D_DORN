#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 18:17
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : ordinal_regression_loss.py
"""

import numpy as np
import torch

import torch.nn.functional as F


class SecondOrdinalRegressionLoss(nn.Module):

    def __init__(self, ord_num, gamma, beta, discretization="SID", double_ord=0):
        super(SecondOrdinalRegressionLoss, self).__init__()
        self.ord_num = ord_num
        self.gamma = gamma ### Minghan: this is missing before 20210307
        self.beta = beta
        self.discretization = discretization

        self.double_ord = double_ord

        mask_local = torch.linspace(0, self.double_ord, self.double_ord+1, requires_grad=False).view(1, -1, 1, 1).float()
        mask_full = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False).view(1, -1, 1, 1).float()

        if self.discretization == "SID":
            log_mask_full = np.log(self.beta) * mask_full / self.ord_num
        else:
            log_mask_full = 1.0 + (self.beta - 1.0) * mask_full / self.ord_num

        self.register_buffer("mask_local", mask_local)
        self.register_buffer("mask_full", mask_full)
        self.register_buffer("log_mask_full", log_mask_full)

    def idx_to_value_topbot(self, ord_idx_top, ord_idx_bot):
        if self.discretization == "SID":
            log_top = np.log(self.beta) * ord_idx_top.float() / self.ord_num
            log_bot = np.log(self.beta) * ord_idx_bot.float() / self.ord_num
        else:
            log_top = 1.0 + (self.beta - 1.0) * ord_idx_top.float() / self.ord_num
            log_bot = 1.0 + (self.beta - 1.0) * ord_idx_bot.float() / self.ord_num
        
        return log_top, log_bot

    def _create_ord_label(self, ord_idx_top, ord_idx_bot, gt):

        log_top, log_bot = self.idx_to_value_topbot(ord_idx_top, ord_idx_bot)
        log_mask_local = log_top + (log_bot - log_top) * self.mask_local / self.double_ord
        log_gt = gt + self.gamma
        if self.discretization == "SID":
            log_gt = torch.log(log_gt)

        ord_c0 = log_gt >= log_mask_local[:, :-1]
        ord_c1 = log_gt < log_mask_local[:, :-1]

        ord_label = torch.cat((ord_c0, ord_c1), dim=1)

        return ord_label

    def interpolate_cdf(self, ord_idx_top, ord_idx_bot, cdf_full, cdf_local):
        
        log_top, log_bot = self.idx_to_value_topbot(ord_idx_top, ord_idx_bot)
        full_in_local = (self.log_mask_full - log_top) / torch.clamp(log_bot-log_top, min=1e-5) * self.double_ord
        full_in_local_floor = full_in_local.floor().clamp(min=0, max=self.double_ord)
        full_in_local_ceil = full_in_local.ceil().clamp(min=0, max=self.double_ord)
        full_in_local_weight_ceil = full_in_local - full_in_local_floor

        cdf_local_01 = torch.cat((cdf_local, torch.zeros_like(cdf_local[:,0])), dim=1)
        cdf_floor = torch.gather(cdf_local_01, 1, full_in_local_floor)
        cdf_ceil = torch.gather(cdf_local_01, 1, full_in_local_ceil)
        cdf_interpolated = cdf_floor * (1-full_in_local_weight_ceil) + cdf_ceil * full_in_local_weight_ceil

        cdf_masked = torch.where((full_in_local>0)&(full_in_local<self.double_ord), cdf_interpolated, cdf_full)

        return cdf_masked

    def forward(self, prob, gt, ord_idx_top, ord_idx_bot):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        if prob.shape != gt.shape:
            prob = F.interpolate(prob, size=gt.shape[-2:], mode="bilinear", align_corners=True)

        # N, C, H, W = prob.shape
        valid_mask = gt > 0.
        gt = torch.unsqueeze(gt, dim=1)
        ord_label = self._create_ord_label(ord_idx_top, ord_idx_bot, gt)
        # print("prob shape: {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask]
        return loss.mean()
