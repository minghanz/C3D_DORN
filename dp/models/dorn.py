#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 21:06
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : dorn.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dp.modules.backbones.resnet import ResNetBackbone
from dp.modules.encoders.SceneUnderstandingModule import SceneUnderstandingModule
from dp.modules.decoders.OrdinalRegression import OrdinalRegressionLayer
from dp.modules.losses.ordinal_regression_loss import OrdinalRegressionLoss

from dp.modules.losses.maxp_loss import MaxPLoss
from dp.modules.losses.neighbor_ctn_depth_loss import NeighborCtnDepthLoss
from dp.modules.losses.second_ordinal_regression_loss import SecondOrdinalRegressionLoss

from dp.modules.decoders.DepthRend import DepthRend

import sys
# sys.path.append("../../../")
from c3d.c3d_loss import C3DLoss
from c3d.utils_general.vis import overlay_dep_on_rgb
import time
# import cv2

class DepthPredModel(nn.Module):

    def __init__(self, ord_num=90, gamma=1.0, beta=80.0,
                 input_size=(385, 513), kernel_size=16, pyramid=[8, 12, 16],
                 batch_norm=False,
                 discretization="SID", pretrained=True, path_of_c3d_cfg=None, 
                 acc_ordreg=False, dyn_weight=False, use_prob_loss=False, feat_dim=32, use_neighbor_depth=False, double_ord=0, render=False, align_corners=True, rend_dropout=0.5):
        super().__init__()
        assert len(input_size) == 2
        assert isinstance(kernel_size, int)
        self.ord_num = ord_num
        self.gamma = gamma
        self.beta = beta
        self.discretization = discretization

        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.SceneUnderstandingModule = SceneUnderstandingModule(ord_num, size=input_size,
                                                                 kernel_size=kernel_size,
                                                                 pyramid=pyramid,
                                                                 batch_norm=batch_norm, 
                                                                 acc_ordreg=acc_ordreg, 
                                                                 dyn_weight=dyn_weight, 
                                                                 feat_dim=feat_dim, 
                                                                 double_ord=double_ord, 
                                                                 align_corners=align_corners)
        self.regression_layer = OrdinalRegressionLayer(acc_ordreg=acc_ordreg, double_ord=double_ord)

        self.flag_use_c3d = path_of_c3d_cfg is not None
        self.flag_use_prob_loss = use_prob_loss
        self.flag_use_neighbor_depth = use_neighbor_depth
        self.flag_double_ord = double_ord > 0
        self.double_ord = double_ord
        self.render = render

        ###Minghan: original dorn loss
        self.criterion = OrdinalRegressionLoss(ord_num, gamma, beta, discretization)
        if self.flag_use_c3d:
            self.criterion2 = C3DLoss(seq_frame_n=1)
            self.criterion2.parse_opts(f_input=path_of_c3d_cfg)
            self.criterion2.cuda() ### ? not sure whether this is needed

        if self.flag_use_prob_loss:
            assert acc_ordreg
            self.criterion_p = MaxPLoss(self.ord_num, self.gamma, self.beta, self.discretization)
            self.criterion_p.cuda()

        if self.flag_use_neighbor_depth:
            assert acc_ordreg
            self.criterion_nd = NeighborCtnDepthLoss(self.ord_num, self.gamma, self.beta, self.discretization)
            self.criterion_nd.cuda()

        if self.flag_double_ord:
            self.criterion_so = SecondOrdinalRegressionLoss(self.ord_num, self.gamma, self.beta, self.discretization, double_ord)

        if self.render:
            self.renderer = DepthRend(n_dep_sample=0, out_class=ord_num, size=input_size, align_corners=align_corners, batch_norm=batch_norm, dropout_prob=rend_dropout)

    def optimizer_params(self):
        group_params = [{"params": filter(lambda p: p.requires_grad, self.backbone.parameters()), 
                         "lr": 1.0},
                        {"params": filter(lambda p: p.requires_grad, self.SceneUnderstandingModule.parameters()),
                         "lr": 10.0}]
        # group_params = [
        #                 {"params": filter(lambda p: p.requires_grad, self.SceneUnderstandingModule.parameters()),
        #                  "lr": 10.0}]
        return group_params

    # def forward(self, image, target=None):
    def forward(self, image, target=None, mask=None, mask_gt=None, cam_info=None):
        """
        :param image: RGB image, torch.Tensor, Nx3xHxW
        :param target: ground truth depth, torch.Tensor, NxHxW
        :return: output: if training, return loss, torch.Float,
                         else return {"target": depth, "prob": prob, "label": label},
                         depth: predicted depth, torch.Tensor, NxHxW
                         prob: probability of each label, torch.Tensor, NxCxHxW, C is number of label
                         label: predicted label, torch.Tensor, NxHxW
        """
        # ### test input correctness
        ### if use this, need to disable "nomalize" in _tr_preprocess in dataset (Kitti.py)
        # test_path = "input_visualization"
        # target_vis = target.clone().detach()
        # target_vis[target_vis < 0] = 0
        # for ib in range(image.shape[0]):
        #     name = "{}.jpg".format(time.time())
        #     overlay_dep_on_rgb(target_vis[ib].unsqueeze(0), image[ib], path=test_path, name=name, overlay=False)

        N, C, H, W = image.shape
        feat = self.backbone(image)
        x_dict = self.SceneUnderstandingModule(feat)
        # print("feat shape:", feat.shape)
        # feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=True)


        # if self.training:
        #     prob = self.regression_layer(feat)
        #     loss = self.criterion(prob, target)
        #     return loss

        # prob, label = self.regression_layer(feat)
        # # print("prob shape:", prob.shape, " label shape:", label.shape)

        # prob_train, prob, label = self.regression_layer(feat)

        if self.render:
            if self.training:
                self.regression_layer(x_dict["full"])
                self.renderer(x_dict, target, mask_gt)
                self.regression_layer(x_dict["samp"])
            else:
                self.renderer(x_dict)
                self.regression_layer(x_dict["full"])
        else:
            self.regression_layer(x_dict["full"])
        
        if self.training:
            loss, losses = self.calc_loss(x_dict["full"], target, mask, mask_gt, cam_info)
            if self.render:
                target_samp = x_dict["samp"]["target"]
                loss_samp, losses_samp = self.calc_loss(x_dict["samp"], target_samp, mask, mask_gt, cam_info)
                loss = loss + loss_samp
                for key in losses:
                    losses[key] = losses[key] + losses_samp[key]
            return loss, losses

        out = x_dict["full"]
        prob = out["pbin"]
        label = out["label"]

        ### visualize entropy
        assert "pmul" in out, "{}".format(list(out.keys()))
        if "pmul" in out:
            pmul = out["pmul"]
            p_bin_clamped = torch.clamp(pmul, min=1e-6, max=1-1e-6)
            entropy = - (p_bin_clamped * torch.log(p_bin_clamped)).sum(1) # B*H*W
            out["entropy"] = entropy

        if self.flag_use_prob_loss:
            depth_pred = (out["pmul"] * self.criterion_p.depth_vector.view(1, -1, 1, 1)).sum(1) # N*H*W
            if self.discretization == "SID":
                depth_pred = torch.exp(depth_pred)
            depth = depth_pred - self.gamma
        elif self.flag_use_neighbor_depth:
            mask_zeros = torch.zeros_like(out["pmul"])
            mask_label = torch.arange(self.ord_num).view(1, -1, 1, 1).to(device=label.device)
            label = label.unsqueeze(1)
            p_bin_masked = torch.where( (mask_label >= label - 2) & (mask_label <= label + 2), out["pmul"], mask_zeros )
            p_bin_masked = F.normalize(p_bin_masked, dim=1, p=1)

            depth_pred = (p_bin_masked * self.criterion_nd.depth_vector.view(1, -1, 1, 1)).sum(1) # N*H*W
            if self.discretization == "SID":
                depth_pred = torch.exp(depth_pred)
            depth = depth_pred - self.gamma
        elif self.flag_double_ord:
            log_top, log_bot = self.criterion_so.idx_to_value_topbot(out["ord_idx_top"], out["ord_idx_bot"])
            log_top = log_top.squeeze(1)
            log_bot = log_bot.squeeze(1)
            t0 = log_top + (log_bot - log_top) * out["label_2"].float() / self.double_ord
            t1 = log_top + (log_bot - log_top) * (out["label_2"].float()+1) / self.double_ord
            depth = (t0 + t1) / 2
            if self.discretization == "SID":
                depth = torch.exp(depth)
            depth = depth - self.gamma
        else:
            if self.discretization == "SID":
                t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
                t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
            else:
                t0 = 1.0 + (self.beta - 1.0) * label.float() / self.ord_num
                t1 = 1.0 + (self.beta - 1.0) * (label.float() + 1) / self.ord_num
            depth = (t0 + t1) / 2 - self.gamma
        # print("depth min:", torch.min(depth), " max:", torch.max(depth),
        #       " label min:", torch.min(label), " max:", torch.max(label))
        out_dict = {"target": [depth], "prob": [prob], "label": [label]}
        if "entropy" in out:
            out_dict["entropy"] = [out["entropy"]]
        return out_dict

    def calc_loss(self, out, target, mask=None, mask_gt=None, cam_info=None):

        prob_train = out["log_pq"]
        prob = out["pbin"]
        label = out["label"]

        losses = dict()
        losses["loss_dorn"] = self.criterion(prob_train, target)
        if self.flag_use_c3d:
            if self.discretization == "SID":
                t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
                t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
            else:
                t0 = 1.0 + (self.beta - 1.0) * label.float() / self.ord_num
                t1 = 1.0 + (self.beta - 1.0) * (label.float() + 1) / self.ord_num
            depth = (t0 + t1) / 2 - self.gamma
            target_bchw = torch.unsqueeze(target, dim=1)
            target_bchw[target_bchw<0] = 0
            depth_bchw = torch.unsqueeze(depth, dim=1)
            # depth_np = depth_bchw.cpu().numpy()
            # gt_np = target_bchw.cpu().numpy()
            # mask_np = mask.cpu().numpy()
            # mask_gt_np = mask_gt.cpu().numpy()
            # print(mask_gt_np.shape)
            # print(mask_np.shape)
            # for i in range(depth_np.shape[0]):
            #     cv2.imwrite("pred_{}.png".format(i), depth_np[i].transpose(1,2,0).astype(np.uint8))
            #     cv2.imwrite("gt_{}.png".format(i), gt_np[i].transpose(1,2,0).astype(np.uint8))
            #     cv2.imwrite("mask_pred_{}.png".format(i), mask_np[i].transpose(1,2,0).astype(np.uint8)*255)
            #     cv2.imwrite("mask_gt_{}.png".format(i), mask_gt_np[i].transpose(1,2,0).astype(np.uint8)*255)

            losses["loss_c3d"] = self.criterion2(image, depth_bchw, target_bchw, mask, mask_gt, cam_info)
        else:
            losses["loss_c3d"] = torch.zeros_like(losses["loss_dorn"])

        if self.flag_use_prob_loss:
            losses_p, depth_pred = self.criterion_p(out["pmul"], target)
            losses["loss_mean"] = losses_p["loss_mean"]
            losses["loss_deviation"] = losses_p["loss_deviation"]
            losses["loss_nll"] = losses_p["loss_nll"]
        elif self.flag_use_neighbor_depth:
            losses_p, depth_pred = self.criterion_nd(out["pmul"], label, target)
            losses["loss_mean"] = losses_p["loss_mean"]
            losses["loss_deviation"] = losses_p["loss_deviation"]
            losses["loss_nll"] = losses_p["loss_nll"]
        elif self.flag_double_ord:
            loss_2nd_dorn = self.criterion_so(out["log_pq_2"], target, out["ord_idx_top"], out["ord_idx_bot"])
            losses["loss_mean"] = loss_2nd_dorn
            losses["loss_deviation"] = torch.zeros_like(losses["loss_dorn"])
            losses["loss_nll"] = torch.zeros_like(losses["loss_dorn"])
        else:
            losses["loss_mean"] = torch.zeros_like(losses["loss_dorn"])
            losses["loss_deviation"] = torch.zeros_like(losses["loss_dorn"])
            losses["loss_nll"] = torch.zeros_like(losses["loss_dorn"])

        loss = losses["loss_dorn"] - 1e-3* losses["loss_c3d"] + losses["loss_mean"] + losses["loss_deviation"] + losses["loss_nll"] ### TODO: weight
        return loss, losses