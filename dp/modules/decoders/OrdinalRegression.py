#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 17:55
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : oridinal_regression_layer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLayer(nn.Module):
    def __init__(self, acc_ordreg=False, double_ord=0):
        """
        acc_ordreg: instead of original DORN regression, we regard each P as the probability of the real value falling in the bin P(j-1<l<j). Then the P(l>j) = sum_1^j(P(j-1<l<j)). 
        """
        super(OrdinalRegressionLayer, self).__init__()

        self.acc_ordreg = acc_ordreg
        self.double_ord = double_ord

    def forward(self, x):
        """
        :param x: NxCxHxW, N is batch_size, C is channels of features
        :return: ord_label is ordinal outputs for each spatial locations , N x 1 x H x W
                 ord prob is the probability of each label, N x OrdNum x H x W
        """
        if self.acc_ordreg:
            return self.forward_acc_ordreg(x)
        else:
            return self.forward_original_DORN(x)

    def forward_acc_ordreg(self, x_dict):
        if self.double_ord > 0:
            assert isinstance(x_dict, dict)
            ord_prob = x_dict["pmul"]
            ord_cdf = x_dict["pbin"]
            ord_prob_2 = x_dict["pmul_local"] 
            ord_cdf_2 = x_dict["pbin_local"]
            N, C, H, W = ord_prob.size()
            ord_num = C
            ord_idx_top = torch.sum((ord_cdf >= 0.7), dim=1, keepdim=True) - 1
            ord_idx_bot = ord_num - torch.sum((ord_cdf <= 0.3), dim=1, keepdim=True)
            ord_idx_top_clamp = ord_idx_top.clamp(min=0, max=ord_num-1)
            ord_idx_bot_clamp = ord_idx_bot.clamp(min=0, max=ord_num-1)
            cdf_max = torch.gather(ord_cdf, 1, ord_idx_top_clamp)
            cdf_min = torch.gather(ord_cdf, 1, ord_idx_bot_clamp)
            ord_cdf_2_scaled = cdf_min + (cdf_max-cdf_min) * ord_cdf_2

            ord_label_2 = torch.sum((ord_cdf_2_scaled > 0.5), dim=1) - 1
        else:
            x = x_dict["logit"]
            N, C, H, W = x.size()
            ord_num = C
            ord_prob = F.softmax(x, dim=1)
            ord_cdf = torch.cumsum(ord_prob.flip([1]), dim=1).flip([1])   # cumsum in the reversed direction
            ord_cdf = F.normalize(ord_cdf, p=float("Inf"), dim=1)

        ord_label = torch.sum((ord_cdf > 0.5), dim=1) - 1

        ord_logp = torch.log(torch.clamp(ord_cdf, min=1e-5))
        ord_logq = torch.log(torch.clamp(1-ord_cdf, min=1e-5))

        ord_log = torch.cat([ord_logp, ord_logq], dim=1)

        x_dict["pmul"] = ord_prob
        x_dict["pbin"] = ord_cdf
        x_dict["log_pq"] = ord_log
        x_dict["label"] = ord_label

        if self.double_ord > 0:
            ord_logp_2 = torch.log(torch.clamp(ord_cdf_2_scaled, min=1e-5))
            ord_logq_2 = torch.log(torch.clamp(1-ord_cdf_2_scaled, min=1e-5))
            ord_log_2 = torch.cat([ord_logp_2, ord_logq_2], dim=1)
            x_dict["log_pq_2"] = ord_log_2
            x_dict["pbin_2"] = ord_cdf_2_scaled
            x_dict["ord_idx_top"] = ord_idx_top
            x_dict["ord_idx_bot"] = ord_idx_bot
            x_dict["label_2"] = ord_label_2

        return #x_dict

        # return ord_log, ord_cdf, ord_label

    def forward_original_DORN(self, x_dict):
        x = x_dict["logit"]
        N, C, H, W = x.size()
        ord_num = C // 2

        # implementation according to the paper
        # A = x[:, ::2, :, :]
        # B = x[:, 1::2, :, :]
        #
        # # A = A.reshape(N, 1, ord_num * H * W)
        # # B = B.reshape(N, 1, ord_num * H * W)
        # A = A.unsqueeze(dim=1)
        # B = B.unsqueeze(dim=1)
        # concat_feats = torch.cat((A, B), dim=1)
        #
        # if self.training:
        #     prob = F.log_softmax(concat_feats, dim=1)
        #     ord_prob = x.clone()
        #     ord_prob[:, 0::2, :, :] = prob[:, 0, :, :, :]
        #     ord_prob[:, 1::2, :, :] = prob[:, 1, :, :, :]
        #     return ord_prob
        #
        # ord_prob = F.softmax(concat_feats, dim=1)[:, 0, ::]
        # ord_label = torch.sum((ord_prob > 0.5), dim=1).reshape((N, 1, H, W))
        # return ord_prob, ord_label

        # reimplementation for fast speed.

        x = x.view(-1, 2, ord_num, H, W)

        ### Minghan: to be able to incorporate other losses, return both nomatter in training mode or not
        # if self.training:
        #     prob = F.log_softmax(x, dim=1).view(N, C, H, W)
        #     return prob

        # ord_prob = F.softmax(x, dim=1)[:, 0, :, :, :]
        # ord_label = torch.sum((ord_prob > 0.5), dim=1)
        # return ord_prob, ord_label

        prob = F.log_softmax(x, dim=1).view(N, C, H, W)
        ord_prob = F.softmax(x, dim=1)[:, 0, :, :, :]
        ord_label = torch.sum((ord_prob > 0.5), dim=1) - 1

        x_dict["log_pq"] = prob
        x_dict["pmul"] = ord_prob
        x_dict["label"] = ord_label
        return #x_dict

        # return prob, ord_prob, ord_label
