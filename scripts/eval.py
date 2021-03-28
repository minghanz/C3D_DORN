#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 22:54
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : train.py
"""

import os
import argparse
import logging
import warnings
import sys
import time
import torch
from tqdm import tqdm
import copy

warnings.filterwarnings("ignore")
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# sys.path.append(os.path.abspath(".."))
sys.path.insert(0, os.path.abspath(".."))  
# sys.path.insert(0, os.path.abspath("../../"))   # for c3d
from c3d.utils_general.eval import eval_preprocess, Metrics
from c3d.utils_general.vis import vis_depth, uint8_np_from_img_tensor, save_np_to_img, vis_normal, uint8_np_from_img_np, overlay_dep_on_rgb, vis_depth_err, overlay_dep_on_rgb_np
from c3d.utils.geometry import NormalFromDepthDense
# from c3d.utils_general.pcl_funcs import pcl_from_grid_xy1_dep, pcl_vis_seq, pcl_write, pcl_load_viewer_fromfile     ## need to install pcl

# running in parent dir
os.chdir("..")

from dp.metircs.average_meter import AverageMeter
from dp.utils.config import load_config, print_config, merge_config
from dp.utils.pyt_io import create_summary_writer
from dp.metircs import build_metrics
from dp.core.solver import Solver
from dp.datasets.loader import build_loader
from dp.visualizers import build_visualizer

from dp.utils.compose_for_eval import compose_preds, kb_crop_preds ### Minghan: This is for evaluating on the full image

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-p', '--pathcfg', type=str)    # separate the path-related config items in another file
parser.add_argument('-r', '--resumed', type=str, default=None, required=False)
parser.add_argument("--local_rank", default=0, type=int)
### options for outputing visualization
parser.add_argument("--vpath", type=str, default="vis")
parser.add_argument("--vpbin", action="store_true", help="visualize probability for each bin")
parser.add_argument("--vent", action="store_true", help="visualize entropy of depth classification and prediction error")
parser.add_argument("--vrgb", action="store_true", help="visualize rgb input images")
parser.add_argument("--vdepth", action="store_true", help="visualize depth gt and predictions")
parser.add_argument("--vmask", action="store_true", help="visualize which pixels are counted in quantitative results")
parser.add_argument("--vrange", action="store_true", help="visualize range mask image which shows rough distance distribution")
parser.add_argument("--vnormal", action="store_true", help="visualize surface normal predictions")
# parser.add_argument("--save-pcl-pred", action="store_true", help="save point cloud prediction to file (need install pcl library)")
# parser.add_argument("--save-pcl-gt", action="store_true", help="save point cloud gt to file (need install pcl library)")
# parser.add_argument("--vpcl", action="store_true", help="visualize pcl as an image (need install pcl library)")
parser.add_argument("--txtpath", type=str, default="eval_result.txt", required=False, help="txt path to save the evaluation result")

args = parser.parse_args()

if not args.config and not args.resumed:
    logging.error('args --config and --resumed should at least one value available.')
    raise ValueError
is_main_process = True if args.local_rank == 0 else False

solver = Solver()

# read config
if args.resumed:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    continue_state_object = torch.load(args.resumed,
                                       map_location=torch.device("cpu"))
    config = continue_state_object['config']
    if args.config:
        override_cfg = load_config(args.config)
        pathcfg = load_config(args.pathcfg)
        override_cfg = merge_config(override_cfg, pathcfg)
        config.update(override_cfg)

    # ### Minghan: only use this line when the trained model and the evaluation is not on the same machine
    # config["data"]["path"] = '/mnt/storage8t/minghanz/Datasets/KITTI_data/kitti_data' 
    # config["data"]["path"] = '/mnt/storage8t/minghanz/Datasets/vKITTI2'  ### for vkitti2 dataset
    ### Minghan: change to the data split you want to test on
    # config["data"]["split"][1] = "eval_visualize"
    # config["data"]["split"][1] = "eval_visualize_vkitti2"
    # ### Minghan: use vkitti2 as evaluation
    # if not isinstance(config["data"]["name"], list):
    #     config["data"]["name"] = [config["data"]["name"]]
    #     config["data"]["name"].append("vKitti2")
    # else:
    #     config["data"]["name"][1] = "vKitti2"
    solver.init_from_checkpoint(continue_state_object=continue_state_object)
    if is_main_process:
        snap_dir = args.resumed[:-len(args.resumed.split('/')[-1])]
        if not os.path.exists(snap_dir):
            logging.error('[Error] {} is not existed.'.format(snap_dir))
            raise FileNotFoundError
else:
    config = load_config(args.config)
    pathcfg = load_config(args.pathcfg)
    config = merge_config(config, pathcfg)

    solver.init_from_scratch(config)
    if is_main_process:
        exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        if isinstance(config['data']['name'], list):
            snap_dir = os.path.join(config["snap"]["path"], config['data']['name'][0],
                                    config['model']['name'], exp_time)
        else:
            snap_dir = os.path.join(config["snap"]["path"], config['data']['name'],
                                    config['model']['name'], exp_time)
        if not os.path.exists(snap_dir):
            os.makedirs(snap_dir)

##### modification of the config for testing
config_left = copy.deepcopy(config)
config_right = copy.deepcopy(config)
config_left["data"]["te_crop_mode"] = "bottom_left"
config_right["data"]["te_crop_mode"] = "bottom_right"

if is_main_process:
    print_config(config)

### output txt file
if args.txtpath is not None:
    assert not os.path.exists(args.txtpath), "The evaluation result txt file already exist: {}".format(args.txtpath)
    txt_folder = os.path.dirname(args.txtpath)
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
        print("The folder for evaluation result txt files created: {}".format(txt_folder))
    else:
        print("The folder for evaluation result txt files exists: {}".format(txt_folder))


# dataset
# tr_loader, sampler, niter_per_epoch = build_loader(config, True, solver.world_size, solver.distributed)
# te_loader, _, niter_test = build_loader(config, False, solver.world_size, solver.distributed)

te_loader_l, _, niter_test_l = build_loader(config_left, False, solver.world_size, solver.distributed)
te_loader_r, _, niter_test_r = build_loader(config_right, False, solver.world_size, solver.distributed)
assert niter_test_l == niter_test_r, "{} {}".format(niter_test_l, niter_test_r)
niter_test = niter_test_l

dataset_name = config["data"]["name"][1]
"""
    usage: debug
"""
# niter_per_epoch, niter_test = 200, 20

loss_meter = AverageMeter()
# metric = build_metrics(config)
### Metrix from c3d
if dataset_name == "Kitti":
    metric = Metrics(d_min=1e-3, d_max=80, shape_unify="kb_crop", eval_crop="garg_crop")
elif dataset_name == "vKitti2":
    metric = Metrics(d_min=1e-3, d_max=80, shape_unify=None, eval_crop="vkitti2", adjust_mean=False, batch_adjust_mean=2.66)  ### for vkitti2
else:
    raise ValueError("data name {} not recognized".format(dataset_name))
mean_tracker = AverageMeter()   ### mean_tracker is to tell whether the scale has any overall shift compared with GT depth
# if is_main_process:
#     writer = create_summary_writer(snap_dir)
#     visualizer = build_visualizer(config, writer)

normal_gener = NormalFromDepthDense()
# if args.vpcl:
#     pcl_viewer = pcl_load_viewer_fromfile()

epoch = config['solver']['epochs']
solver.after_epoch()

header_row = '{:10s} {} {:6s} {:6s} {:6s}'.format("Iter", metric.get_header_row(), "IO", "Inf", "Cmp")
print(header_row)
row_fmt = '{:10s} {} {:6.2f} {:6.2f} {:6.2f}'

# validation
if is_main_process:
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(niter_test), file=sys.stdout, bar_format=bar_format)
else:
    pbar = range(niter_test)
metric.reset()
test_iter_l = iter(te_loader_l)
test_iter_r = iter(te_loader_r)
for idx in pbar:
    t_start = time.time()
    minibatch_l = test_iter_l.next()
    filtered_kwargs_l = solver.parse_kwargs(minibatch_l)
    minibatch_r = test_iter_r.next()
    filtered_kwargs_r = solver.parse_kwargs(minibatch_r)

    # print(filtered_kwargs)
    t_end = time.time()
    io_time = t_end - t_start
    t_start = time.time()
    pred_l = solver.step_no_grad(**filtered_kwargs_l)
    pred_r = solver.step_no_grad(**filtered_kwargs_r)
    d_pred_l = pred_l["target"][-1] # B*H*W
    d_pred_r = pred_r["target"][-1] # B*H*W

    full_width = minibatch_l["depth_full"].shape[-1] # minibatch_l["depth_full"].shape=(B*H*W)
    full_height = minibatch_l["depth_full"].shape[-2]
    pred_full = compose_preds(d_pred_l, d_pred_r, full_width, full_height)
    pred_kb_crop = kb_crop_preds(pred_full) # B*H*W

    t_end = time.time()
    inf_time = t_end - t_start

    t_start = time.time()
    ### remove batch dim
    pred_crop_hw = pred_kb_crop[0]
    pred_full_hw = pred_full[0]
    gt_full_hw = minibatch_l["depth_full"][0]
    if dataset_name == "Kitti":
        extra_dict = metric.compute_metric(pred_crop_hw, gt_full_hw)
    elif dataset_name == "vKitti2":
        extra_dict = metric.compute_metric(pred_full_hw, gt_full_hw)     # for vkitti2
    else:
        raise ValueError("data name {} not recognized".format(dataset_name))
    t_end = time.time()
    cmp_time = t_end - t_start

    mean_tracker.update(extra_dict['mean_pred']/extra_dict['mean_gt'])

    ### entropy and prediction error
    if args.vent:
        img_full = minibatch_l["image_full"][0]
        img_full_np = uint8_np_from_img_tensor(img_full)

        entropy_l = pred_l["entropy"][-1]
        entropy_r = pred_r["entropy"][-1]
        entropy_full_hw = compose_preds(entropy_l, entropy_r, full_width, full_height)[0]
        
        dep_err = vis_depth_err(pred_full_hw.unsqueeze(0), gt_full_hw.unsqueeze(0))
        entropy_full_hw = entropy_full_hw.unsqueeze(0)

        dep_err_np = overlay_dep_on_rgb_np(dep_err, img_full_np, overlay=True)
        entropy_img_np = overlay_dep_on_rgb(entropy_full_hw, img_full, overlay=False)
        entropy_np = overlay_dep_on_rgb(entropy_full_hw, img_full, overlay=True)
        save_np_to_img(dep_err_np, "{}/{}_pdep_err".format(args.vpath, idx))
        save_np_to_img(entropy_img_np, "{}/{}_pent_img".format(args.vpath, idx))
        save_np_to_img(entropy_np, "{}/{}_pent".format(args.vpath, idx))
    

    #################################################### visualization
    if args.vpbin:
        p_cdf = pred_l["prob"][0]  # ord_num*H*W
        p_cdf_slide = p_cdf[..., 100]   # ord_num*H
        p_cdf_slide = p_cdf_slide.expand(3, -1, -1) # 3*ord_num*H
        p_cdf_np = uint8_np_from_img_tensor(p_cdf_slide)
        save_np_to_img(p_cdf_np, "{}/{}_pbin_acc".format(args.vpath, idx))

    if args.vmask:
        ### visualize mask showing pixels included in quantitative result
        mask = extra_dict['mask']
        img_mask = uint8_np_from_img_tensor(mask.unsqueeze(0))
        save_np_to_img(img_mask, "{}/{}_mask".format(args.vpath, idx))

    if args.vrange:
        ## visualize range-indicating mask of ground truth and predictions, again it is to tell whether the scale has any overall shift compared with GT depth
        gt_5 = gt_full_hw <= 5
        gt_10 = gt_full_hw <= 10
        gt_20 = gt_full_hw <= 20
        gt_51020 = torch.stack([gt_5, gt_10, gt_20], 0)
        img_mask = uint8_np_from_img_tensor(gt_51020)
        save_np_to_img(img_mask, "{}/{}_mask_gt51020".format(args.vpath, idx))
        
        gt_5 = pred_full <= 5
        gt_10 = pred_full <= 10
        gt_20 = pred_full <= 20
        gt_51020 = torch.cat([gt_5, gt_10, gt_20], 0)
        img_mask = uint8_np_from_img_tensor(gt_51020)
        save_np_to_img(img_mask, "{}/{}_mask_pred51020".format(args.vpath, idx))

    if args.vrgb:
        ### visualize rgb image
        img_full = minibatch_l["image_full"][0]
        img_kb_crop = kb_crop_preds(img_full)
        img_full_np = uint8_np_from_img_tensor(img_full)
        save_np_to_img(img_full_np, "{}/{}_rgb".format(args.vpath, idx))

    if args.vdepth:
        ### visualize predictions
        vis_pred = vis_depth(pred_full)
        vis_pred = uint8_np_from_img_tensor(vis_pred)
        vis_gt = vis_depth(minibatch_l["depth_full"])
        vis_gt = uint8_np_from_img_tensor(vis_gt)
        save_np_to_img(vis_pred, "{}/{}_pred".format(args.vpath, idx))
        save_np_to_img(vis_gt, "{}/{}_gt".format(args.vpath, idx))

    if args.vnormal:
        ### generate normal image
        # cam_info_kb = minibatch_l["cam_info_kb_crop"]
        # normal_pred = normal_gener(pred_kb_crop.unsqueeze(1), cam_info_kb.K)
        cam_info_full = minibatch_l["cam_info_full"]
        normal_pred = normal_gener(pred_full.unsqueeze(1), cam_info_full.K)
        vis_normal_pred = vis_normal(normal_pred)
        vis_normal_pred = uint8_np_from_img_tensor(vis_normal_pred)
        save_np_to_img(vis_normal_pred, "{}/{}_normal".format(args.vpath, idx))

    # if args.save_pcl_pred or args.vpcl:
    #     ### generate pcl (saving snapshot need a GUI environment)
    #     xy1_pred = minibatch_l["cam_info_full"].xy1_grid
    #     pcd_pred = pcl_from_grid_xy1_dep(xy1_pred, pred_full, minibatch_l['image_full'])
    #     if args.save_pcl_pred:
    #         pcl_write(pcd_pred[0], "{}/{}_pred".format(args.vpath, idx))
    #     if args.vpcl:
    #         pcl_vis_seq(pcd_pred, viewer=pcl_viewer, snapshot_fname_fmt="{}/{}_pcdvis_pred".format(args.vpath, idx)+"_{}")

    # if args.save_pcl_gt or args.vpcl:
    #     xy1_pred = minibatch_l["cam_info_full"].xy1_grid
    #     pcd_gt = pcl_from_grid_xy1_dep(xy1_pred, minibatch_l['depth_full'], minibatch_l['image_full'])
    #     if args.save_pcl_gt:
    #         pcl_write(pcd_gt[0], "{}/{}_gt".format(args.vpath, idx))
    #     if args.vpcl:
    #         pcl_vis_seq(pcd_gt, viewer=pcl_viewer, snapshot_fname_fmt="{}/{}_pcdvis_gt".format(args.vpath, idx)+"_{}")

    ##########################################################################

    if is_main_process:
        # print_str = '[Test] Epoch{}/{}'.format(epoch, config['solver']['epochs']) \
        #             + ' Iter{}/{}: '.format(idx + 1, niter_test) \
        #             + metric.get_snapshot_info() \
        #             + ' IO:%.2f' % io_time \
        #             + ' Inf:%.2f' % inf_time \
        #             + ' Cmp:%.2f' % cmp_time

        print_str = row_fmt.format("{}/{}".format(idx+1, niter_test), metric.get_snapshot_row(), io_time, inf_time, cmp_time)
        pbar.set_description(print_str, refresh=False)

    
if is_main_process:
    print(metric.get_header_row())
    print(metric.get_result_row())
    print("Mean pred/gt ratio:", mean_tracker.mean())

    if args.txtpath is not None:
        assert not os.path.exists(args.txtpath), args.txtpath
        with open(args.txtpath, "w") as f:
            print(metric.get_header_row(), file=f)
            print(metric.get_result_row(), file=f)
            print("Mean pred/gt ratio: {}".format(mean_tracker.mean()), file=f)
            

# if is_main_process:
#     logging.info('After Epoch{}/{}, {}'.format(epoch, config['solver']['epochs'], metric.get_result_info()))
#     # writer.add_scalar("Train/loss", loss_meter.mean(), epoch)
#     metric.add_scalar(writer, tag='Test', epoch=epoch)