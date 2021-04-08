# -*- coding: utf-8 -*-
# @Time    : 2019/12/31 下午3:04
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : visualization.py


import numpy as np
import matplotlib.pyplot as plt


def depth_to_color(depth, rgb=True, fix_min=None, fix_max=None):
    if rgb:
        cmap = plt.cm.jet     # output rgb map
    else:
        cmap = plt.cm.Greys   

    if fix_min is not None:
        d_min = fix_min
    else:
        d_min = np.min(depth)

    if fix_max is not None:
        d_max = fix_max
    else:
        d_max = np.max(depth)
        
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth_relative = np.clip(depth_relative, 0, 1)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def error_to_color(depth, gt, rgb=False, fix_min=None, fix_max=None):
    mask = gt <= 0.
    if rgb:
        cmap = plt.cm.jet     # output rgb map
    else:
        cmap = plt.cm.Greys     # output gray map
    err = np.abs(depth-gt)
    err[mask] = 0.

    if fix_min is not None:
        err_min = fix_min
    else:
        err_min = np.min(err)

    if fix_max is not None:
        err_max = fix_max
    else:
        err_max = np.max(err)

    err_rel = (err-err_min) / (err_max-err_min)
    err_rel = np.clip(err_rel, 0, 1)
    return 255 * cmap(err_rel)[:, :, :3]

