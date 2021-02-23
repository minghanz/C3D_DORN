#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-07-17
@Author  : Minghan Zhu
@Email   : minghanz@umich.edu
@File    : Kitti_raw_lidar.py
"""
"""
This file creates a dataset taking raw lidar scan as source of depth gt 
"""

import os
import cv2
import math
import random
import numpy as np


from PIL import Image

from dp.datasets.base_dataset import BaseDataset
from dp.datasets.utils import nomalize, PILLoader, KittiDepthLoader
from dp.datasets.Kitti import Kitti

import sys
# sys.path.append("../../../")
from c3d.utils_general.dataset_read import DataReaderKITTI
from c3d.utils_general.calib import lidar_to_depth
from c3d.utils.cam_proj import CamProj, seq_ops_on_cam_info
from c3d.utils.cam import CamCrop

from skimage.morphology import binary_dilation, binary_closing

import torch

class KittiRawLidar(Kitti):

    def __init__(self, config, is_train=True, image_loader=PILLoader, depth_loader=None):
        super().__init__(config, is_train, image_loader, depth_loader)

        self.dilate_struct = np.ones((35, 35))
    
    def depth_loader(self, file):
        # loads depth map D from lidar file
        ### This function is the same as in bts_dataloader __getitem__
        assert os.path.exists(file), "file not found: {}".format(file)
        wanted_ftype_list = ['calib', 'lidar']
        ntp = self.datareader.ffinder.ntp_from_fname(file, 'lidar')
        data_dict = self.datareader.read_datadict_from_ntp(ntp, wanted_ftype_list)
        velo = data_dict['lidar']
        # K_unit = data_dict['calib'].K_unit
        # extr_cam_li = data_dict['calib'].P_cam_li
        # im_shape = (data_dict['calib'].height, data_dict['calib'].width)
        # depth_gt = lidar_to_depth(velo, extr_cam_li, K_unit, im_shape, torch_mode=False)

        in_ex = data_dict['calib']
        depth_gt = in_ex.lidar_to_depth(velo)
        
        depth_gt[depth_gt == 0] = -1. ## this is to be consistent with KittiDepthLoader in utils.py
        return depth_gt

    def _tr_preprocess(self, image, depth, image_path):
        ### Minghan: load cam_info, which should be adjusted with preprocessing logged in cam_ops
        ntp = self.datareader.ffinder.ntp_from_fname(image_path, 'rgb')
        if 'calib' in self.datareader.ffinder.preload_ftypes:
            ntp_cam = self.cam_proj.dataset_reader.ffinder.ntp_ftype_convert(ntp, ftype='calib')
            cam_info = self.cam_proj.prepare_cam_info(key=ntp_cam)
        else:
            inex = self.datareader.read_from_ntp(ntp, ftype='calib')
            cam_info = self.cam_proj.prepare_cam_info(intr=inex)
            
        cam_ops = []

        crop_h, crop_w = self.config["tr_crop_size"]
        # resize
        W, H = image.size
        dH, dW = depth.shape

        ### Minghan: filled depth is of the same size, which may not be the same as images
        minW = W if W < dW else dW
        minH = H if H < dH else dH
        # assert W == dW and H == dH, \
        #     "image shape should be same with depth, but image shape is {}, depth shape is {}".format((H, W), (dH, dW))

        # scale_h, scale_w = max(crop_h/H, 1.0), max(crop_w/W, 1.0)
        # scale = max(scale_h, scale_w)
        # H, W = math.ceil(scale*H), math.ceil(scale*W)
        # H, W = max(int(scale*H), crop_h), max(int(scale*W), crop_w)

        # print("w={}, h={}".format(W, H))
        scale = max(crop_h / H, 1.0)
        H, W = max(crop_h, H), math.ceil(scale * W)
        image = image.resize((W, H), Image.BILINEAR)
        # depth = cv2.resize(depth, (W, H), cv2.INTER_LINEAR)
        # print("image shape:", image.size, " depth shape:", depth.shape)

        crop_dh, crop_dw = int(crop_h / scale), int(crop_w / scale)

        # random crop size
        ### Minghan: in case image and depth are not of the same size
        # x = random.randint(0, W - crop_w)
        # y = random.randint(0, H - crop_h)
        x = random.randint(0, minW - crop_w)
        y = random.randint(0, minH - crop_h)
        
        dx, dy = math.floor(x/scale), math.floor(y/scale)
        # print("corp dh = {}, crop dw = {}".format(crop_dh, crop_dw))

        image = image.crop((x, y, x + crop_w, y + crop_h))
        depth = depth[dy:dy + crop_dh, dx:dx + crop_dw]
        # print("depth shape: ", depth.shape)

        ### Minghan: log the cropping operation in cam_ops
        cam_ops.append(CamCrop(x, y, crop_w, crop_h))
        ### Minghan: we assume always using scale = 1
        assert crop_h == crop_dh
        assert crop_w == crop_dw
        assert scale == 1
        assert x == dx
        assert y == dy
        

        # normalize
        image = np.asarray(image).astype(np.float32) / 255.0
        image = nomalize(image, type=self.config['norm_type'])
        image = image.transpose(2, 0, 1)

        ### mask used by c3d loss
        mask_gt = depth > 0
        mask = binary_closing(mask_gt, self.dilate_struct)
        ## create channel dimension at the end (does not expand depth here to be consistent with original dorn)
        mask_gt = np.expand_dims(mask_gt, axis=0)
        mask = np.expand_dims(mask, axis=0)
        ### Minghan: update cam_info according to cam_ops
        cam_info = seq_ops_on_cam_info(cam_info, cam_ops)
        extra_dict = {"cam_info": cam_info, "mask": mask, "mask_gt": mask_gt }

        return image, depth, extra_dict