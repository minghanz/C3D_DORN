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

import sys
sys.path.append("../../../")
from c3d.c3d_loss import C3DLoss
from c3d.utils_general.vis import overlay_dep_on_rgb
import time
# import cv2

class DepthPredModel(nn.Module):

    def __init__(self, ord_num=90, gamma=1.0, beta=80.0,
                 input_size=(385, 513), kernel_size=16, pyramid=[8, 12, 16],
                 batch_norm=False,
                 discretization="SID", pretrained=True, path_of_c3d_cfg=None):
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
                                                                 batch_norm=batch_norm)
        self.regression_layer = OrdinalRegressionLayer()

        self.flag_use_c3d =path_of_c3d_cfg is not None
        if not self.flag_use_c3d:
            ###Minghan: original dorn loss
            self.criterion = OrdinalRegressionLoss(ord_num, beta, discretization)
        else:
            self.criterion = OrdinalRegressionLoss(ord_num, beta, discretization)
            self.criterion2 = C3DLoss()
            self.criterion2.parse_opts(f_input=path_of_c3d_cfg)
            self.criterion2.cuda() ### ? not sure whether this is needed

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
        feat = self.SceneUnderstandingModule(feat)
        # print("feat shape:", feat.shape)
        # feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=True)


        # if self.training:
        #     prob = self.regression_layer(feat)
        #     loss = self.criterion(prob, target)
        #     return loss

        # prob, label = self.regression_layer(feat)
        # # print("prob shape:", prob.shape, " label shape:", label.shape)

        prob_train, prob, label = self.regression_layer(feat)
        if self.training:
            loss_dorn = self.criterion(prob_train, target)
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

                loss_c3d = self.criterion2(image, depth_bchw, target_bchw, mask, mask_gt, cam_info)
                loss = loss_dorn - 1e-3* loss_c3d ### TODO: weight
                return loss, loss_dorn, loss_c3d
            else:
                return loss_dorn, loss_dorn, torch.zeros_like(loss_dorn)
            return loss

        if self.discretization == "SID":
            t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
            t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
        else:
            t0 = 1.0 + (self.beta - 1.0) * label.float() / self.ord_num
            t1 = 1.0 + (self.beta - 1.0) * (label.float() + 1) / self.ord_num
        depth = (t0 + t1) / 2 - self.gamma
        # print("depth min:", torch.min(depth), " max:", torch.max(depth),
        #       " label min:", torch.min(label), " max:", torch.max(label))
        return {"target": [depth], "prob": [prob], "label": [label]}
