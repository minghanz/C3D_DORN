#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 22:28
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : Kitti.py
"""

import os
import cv2
import math
import random
import numpy as np


from PIL import Image

from dp.datasets.base_dataset import BaseDataset
from dp.datasets.utils import nomalize, PILLoader, KittiDepthLoader

import sys
sys.path.append("../../../")
from c3d.utils_general.dataset_read import DataReaderKITTI, DataReaderVKITTI2
from c3d.utils.cam_proj import CamProj, seq_ops_on_cam_info
from c3d.utils.cam import CamCrop

import torch

class Kitti(BaseDataset):

    def __init__(self, config, is_train=True, image_loader=PILLoader, depth_loader=KittiDepthLoader):
        super().__init__(config, is_train, image_loader, depth_loader)
        # file_list = "./dp/datasets/lists/kitti_{}.list".format(self.split)
        file_list = "./dp/datasets/lists/kitti_{}_from_bts.list".format(self.split)
        with open(file_list, "r") as f:
            self.filenames = f.readlines()
            ### skip the files with no ground truth
            self.filenames = [x for x in self.filenames if "None" not in x]

        self.init_c3d()

    def init_c3d(self):
        self.datareader = DataReaderKITTI(data_root=self.root)
        self.cam_proj = CamProj(self.datareader, batch_size=1)

    def _parse_path(self, index):
        image_path, depth_path = self.filenames[index].split()
        image_path = os.path.join(self.root, image_path)
        if depth_path != "None":
            depth_path = os.path.join(self.root, depth_path)
        else:
            depth_path = None
        return image_path, depth_path

    def __getitem__(self, index):
        image_path, depth_path = self._parse_path(index)
        item_name = image_path.split("/")[-1].split(".")[0]

        image, depth = self._fetch_data(image_path, depth_path)
        image, depth, extra_dict = self.preprocess(image, depth, image_path)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        output_dict = dict(image=image,
                           fn=str(item_name),
                           image_path=image_path,
                           n=self.get_length())

        if depth is not None:
            output_dict['target'] = torch.from_numpy(np.ascontiguousarray(depth)).float()
            output_dict['target_path'] = depth_path

        if extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _tr_preprocess(self, image, depth, image_path):
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

        # normalize
        image = np.asarray(image).astype(np.float32) / 255.0
        image = nomalize(image, type=self.config['norm_type'])
        image = image.transpose(2, 0, 1)

        return image, depth, None

    def _te_preprocess(self, image, depth, image_path):
        ### Minghan: load cam_info, which should be adjusted with preprocessing logged in cam_ops
        ntp = self.datareader.ffinder.ntp_from_fname(image_path, 'rgb')
        if 'calib' in self.datareader.ffinder.preload_ftypes:
            ntp_cam = self.cam_proj.dataset_reader.ffinder.ntp_ftype_convert(ntp, ftype='calib')
            cam_info = self.cam_proj.prepare_cam_info(key=ntp_cam)
        else:
            inex = self.datareader.read_from_ntp(ntp, ftype='calib')
            cam_info = self.cam_proj.prepare_cam_info(intr=inex)
        cam_ops = []

        ### for evaluating on full image
        ### Minghan: this is only applicable if batch_size=1 for evaluation, because raw full images in KITTI are not of exactly the same size
        depth_full = depth.copy() if depth is not None else None
        image_full = image.copy()
        image_full = np.array(image_full).astype(np.float32)
        image_full = image_full.transpose(2, 0, 1)

        crop_h, crop_w = self.config["te_crop_size"]
        # resize
        W, H = image.size
        if depth is not None:
            dH, dW = depth.shape
        else:
            dH = H
            dW = W

        assert W == dW and H == dH, \
            "image shape should be same with depth, but image shape is {}, depth shape is {}".format((H, W), (dH, dW))       

        # scale_h, scale_w = max(crop_h/H, 1.0), max(crop_w/W, 1.0)
        # scale = max(scale_h, scale_w)
        # # H, W = math.ceil(scale*H), math.ceil(scale*W)
        scale = max(crop_h / H, 1.0)
        H, W = max(crop_h, H), math.ceil(scale * W)
        # H, W = max(int(scale*H), crop_h), max(int(scale*W), crop_w)

        image_n = image.copy()
        image = image.resize((W, H), Image.BILINEAR)
        crop_dh, crop_dw = int(crop_h/scale), int(crop_w/scale)
        # print("corp dh = {}, crop dw = {}".format(crop_dh, crop_dw))
        # depth = cv2.resize(depth, (W, H), cv2.INTER_LINEAR)

        assert crop_dh == crop_h, "{} {}".format(crop_dh, crop_h)
        assert crop_dw == crop_w, "{} {}".format(crop_dw, crop_w)
        crop_mode = self.config["te_crop_mode"]
        if crop_mode == "center":
            # center crop
            x = (W - crop_w) // 2
            y = (H - crop_h) // 2
            dx = (dW - crop_dw) // 2
            dy = (dH - crop_dh) // 2
        elif crop_mode == "kb_crop":
            ### this mode actually cannot be used because DORN requires fixed size input due to fc layers
            assert crop_w == 1216, crop_w
            assert crop_h == 352, crop_h
            
            x = (W - crop_w) // 2
            y = H - crop_h
            dx = (dW - crop_dw) // 2
            dy = dH - crop_dh
        elif crop_mode == "bottom_left":
            x = 0
            y = H - crop_h
            dx = 0
            dy = dH - crop_dh
        elif crop_mode == "bottom_right":
            x = W - crop_w
            y = H - crop_h
            dx = dW - crop_dw
            dy = dH - crop_dh
        elif crop_mode == "random":
            x = random.randint(0, W - crop_w)
            y = random.randint(0, H - crop_h)
            dx = random.randint(0, dW - crop_dw)
            dy = random.randint(0, dH - crop_dh)
        else: 
            raise ValueError("crop_mode {} not recognized".format(crop_mode))


        image = image.crop((x, y, x + crop_w, y + crop_h))
        if depth is not None:
            depth = depth[dy:dy + crop_dh, dx:dx + crop_dw]
        image_n = image_n.crop((dx, dy, dx + crop_dw, dy + crop_dh))

        # normalize
        image_n = np.array(image_n).astype(np.float32)
        image = np.asarray(image).astype(np.float32) / 255.0
        image = nomalize(image, type=self.config['norm_type'])
        image = image.transpose(2, 0, 1)

        output_dict = {"image_n": image_n}

        ### Minghan: log the cropping operation in cam_ops
        cam_ops.append(CamCrop(x, y, crop_w, crop_h))
        ### Minghan: we assume always using scale = 1
        assert crop_h == crop_dh
        assert crop_w == crop_dw
        assert scale == 1
        assert x == dx
        assert y == dy

        cam_info_img = seq_ops_on_cam_info(cam_info, cam_ops)
        x_kb = (W - 1216) // 2
        y_kb = H - 352
        cam_info_kb_crop = seq_ops_on_cam_info(cam_info, [CamCrop(x_kb, y_kb, 1216, 352)] )

        ### Minghan: save full image for future reference
        ### Minghan: note that "cam_info_full" is likely not able to be batched because the shape of full images are not the same. 
        ###          However the testing dataloader is likely to have batch size 1 so it is okay.
        output_dict.update({"cam_info": cam_info_img, "cam_info_kb_crop": cam_info_kb_crop, "cam_info_full": cam_info, "depth_full": depth_full, "image_full": image_full})

        return image, depth, output_dict
