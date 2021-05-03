#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 18:33
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : dorn.py
"""
import numpy as np

from dp.visualizers.utils import depth_to_color, error_to_color, normalize_to_01, scalar_01_to_color
from dp.visualizers.base_visualizer import BaseVisualizer
from dp.utils.pyt_ops import tensor2numpy, interpolate


class dorn_visualizer(BaseVisualizer):
    def __init__(self, config, writer=None):
        super(dorn_visualizer, self).__init__(config, writer)

    def visualize(self, batch, out, epoch=0, idx=0):
        """
            :param batch_in: minibatch
            :param pred_out: model output for visualization, dic, {"target": [NxHxW]}
            :param tensorboard: if tensorboard = True, the visualized image should be in [0, 1].
            :return: vis_ims: image for visualization.
            """
        fn = batch["fn"]
        idx = "%02d"%idx
        if batch["target"].shape != out["target"][-1].shape:
            h, w = batch["target"].shape[-2:]
            # batch = interpolate(batch, size=(h, w), mode='nearest')
            out = interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        image = batch["image_n"].numpy()

        has_gt = False
        if batch.get("target") is not None:
            depth_gts = tensor2numpy(batch["target"])
            has_gt = True

        for i in range(len(fn)):
            image = image[i].astype(np.float)
            depth = tensor2numpy(out['target'][0][i])
            # print("!! depth shape:", depth.shape)

            if has_gt:
                depth_gt = depth_gts[i]

                err = error_to_color(depth, depth_gt, rgb=True, fix_min=0, fix_max=10)
                depth_gt = depth_to_color(depth_gt)

            depth = depth_to_color(depth)
            # print("pred:", depth.shape, " target:", depth_gt.shape)

            ### entropy:
            if out.get("entropy") is not None:
                entropy = tensor2numpy(out['entropy'][0][i])
                entropy = depth_to_color(entropy, rgb=True, fix_min=0, fix_max=1)
                group = np.concatenate((entropy, depth), axis=0)
            elif out.get("feat_painter") is not None:
                feat_painter = tensor2numpy(out['feat_painter'][0][i])
                for cn in range(3):
                    feat_painter[cn] = normalize_to_01(feat_painter[cn]) * 255
                feat_painter = feat_painter.transpose(1, 2, 0)
                # print("feat_painter.shape", feat_painter.shape)
                # print("depth.shape", depth.shape)
                group = np.concatenate((feat_painter, depth), axis=0)
            else:
                group = np.concatenate((image, depth), axis=0)

            if has_gt:
                if out.get("sample_painter") is not None:
                    sample_painter = tensor2numpy(out['sample_painter'][0][i])  # H*W
                    sample_painter = depth_to_color(sample_painter, rgb=False, fix_min=0, fix_max=1)
                    gt_group = np.concatenate((sample_painter, err), axis=0)
                else:
                    gt_group = np.concatenate((depth_gt, err), axis=0)

                group = np.concatenate((group, gt_group), axis=1)

            if self.writer is not None:
                group = group.transpose((2, 0, 1)) / 255.0
                group = group.astype(np.float32)
                # print("group shape:", group.shape)
                self.writer.add_image(idx+"_"+fn[i] + "/image", group, epoch)