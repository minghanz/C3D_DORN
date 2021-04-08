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


from dp.datasets.Kitti import Kitti

import sys
# sys.path.append("../../../")
from c3d.utils_general.dataset_read import DataReaderVKITTI2
from c3d.utils.cam_proj import CamProj

from dp.datasets.utils import nomalize, PILLoader, vKitti2DepthLoader

class vKitti2(Kitti):
    
    def __init__(self, config, is_train=True, image_loader=PILLoader, depth_loader=None):
        super().__init__(config, is_train, image_loader, depth_loader)

        self.depth_loader = vKitti2DepthLoader(self.datareader)

    def init_c3d(self):
        self.datareader = DataReaderVKITTI2(data_root=self.root)
        self.cam_proj = CamProj(self.datareader, batch_size=1)