# -*- coding: utf-8 -*-
# @Time    : 2019/12/31 下午3:04
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : visualization.py


import numpy as np
import matplotlib.pyplot as plt


def depth_to_color(depth, rgb=True, fix_min=None, fix_max=None):  

    depth_relative = normalize_to_01(depth, fix_min, fix_max)
    return scalar_01_to_color(depth_relative, rgb)


def error_to_color(depth, gt, rgb=False, fix_min=None, fix_max=None):
    mask = gt <= 0.
    err = np.abs(depth-gt)
    err[mask] = 0.

    err_rel = normalize_to_01(err, fix_min, fix_max)
    return scalar_01_to_color(err_rel, rgb)

def normalize_to_01(value, fix_min=None, fix_max=None):
    
    if fix_min is not None:
        d_min = fix_min
    else:
        d_min = np.min(value)

    if fix_max is not None:
        d_max = fix_max
    else:
        d_max = np.max(value)
        
    value_relative = (value - d_min) / (d_max - d_min)
    value_relative = np.clip(value_relative, 0, 1)

    return value_relative

def scalar_01_to_color(value_rel, rgb=True):
    if rgb:
        cmap = plt.cm.jet     # output rgb map
    else:
        cmap = plt.cm.Greys     # output gray map

    return 255 * cmap(value_rel)[:, :, :3]  # H, W, C