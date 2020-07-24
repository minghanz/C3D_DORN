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
sys.path.insert(0, os.path.abspath("../../"))   # for c3d
from c3d.utils_general.eval import eval_preprocess, Metrics
from c3d.utils_general.vis import vis_depth, uint8_np_from_img_tensor, save_np_to_img, vis_normal
from c3d.utils.geometry import NormalFromDepthDense

# running in parent dir
os.chdir("..")

from dp.metircs.average_meter import AverageMeter
from dp.utils.config import load_config, print_config
from dp.utils.pyt_io import create_summary_writer
from dp.metircs import build_metrics
from dp.core.solver import Solver
from dp.datasets.loader import build_loader
from dp.visualizers import build_visualizer

from dp.utils.compose_for_eval import compose_preds, kb_crop_preds ### Minghan: This is for evaluating on the full image

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-r', '--resumed', type=str, default=None, required=False)
parser.add_argument("--local_rank", default=0, type=int)

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
    solver.init_from_checkpoint(continue_state_object=continue_state_object)
    if is_main_process:
        snap_dir = args.resumed[:-len(args.resumed.split('/')[-1])]
        if not os.path.exists(snap_dir):
            logging.error('[Error] {} is not existed.'.format(snap_dir))
            raise FileNotFoundError
else:
    config = load_config(args.config)
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

# dataset
# tr_loader, sampler, niter_per_epoch = build_loader(config, True, solver.world_size, solver.distributed)
# te_loader, _, niter_test = build_loader(config, False, solver.world_size, solver.distributed)

te_loader_l, _, niter_test_l = build_loader(config_left, False, solver.world_size, solver.distributed)
te_loader_r, _, niter_test_r = build_loader(config_right, False, solver.world_size, solver.distributed)
assert niter_test_l == niter_test_r, "{} {}".format(niter_test_l, niter_test_r)
niter_test = niter_test_l

"""
    usage: debug
"""
# niter_per_epoch, niter_test = 200, 20

loss_meter = AverageMeter()
# metric = build_metrics(config)
### Metrix from c3d
metric = Metrics(d_min=1e-3, d_max=80, shape_unify="kb_crop", eval_crop="garg_crop")
# if is_main_process:
#     writer = create_summary_writer(snap_dir)
#     visualizer = build_visualizer(config, writer)

normal_gener = NormalFromDepthDense()

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
    pred_full = compose_preds(d_pred_l, d_pred_r, full_width)
    pred_kb_crop = kb_crop_preds(pred_full) # B*H*W

    t_end = time.time()
    inf_time = t_end - t_start

    t_start = time.time()
    # metric.compute_metric(pred, filtered_kwargs)
    # ### Minghan: use postprocessed items to be strictly aligned with quantitative definition
    # pred_to_eval = dict()
    # pred_to_eval["target"] = []
    # gt_to_eval = dict()
    # gt_to_eval["target"] = []
    # for ib in range(pred_kb_crop.shape[0]):
    #     depth_pred_masked, depth_gt_masked = eval_preprocess(pred_kb_crop[ib], minibatch_l["depth_full"][ib], d_min=1e-3, d_max=80, shape_unify="kb_crop", eval_crop="garg_crop")
    #     pred_to_eval["target"].append(depth_pred_masked)
    #     gt_to_eval["target"].append(depth_gt_masked)
    # pred_to_eval["target"] = [torch.stack(pred_to_eval["target"], dim=0)]
    # gt_to_eval["target"] = torch.stack(gt_to_eval["target"], dim=0)
    # metric.compute_metric(pred_to_eval, gt_to_eval)
    pred_crop_hw = pred_kb_crop[0]
    gt_full_hw = minibatch_l["depth_full"][0]
    metric.compute_metric(pred_crop_hw, gt_full_hw)
    t_end = time.time()
    cmp_time = t_end - t_start

    ### visualize predictions
    vis_pred = vis_depth(pred_kb_crop)
    vis_pred = uint8_np_from_img_tensor(vis_pred)
    vis_gt = vis_depth(minibatch_l["depth_full"])
    vis_gt = uint8_np_from_img_tensor(vis_gt)
    save_np_to_img(vis_pred, "vis/{}_pred".format(idx))
    save_np_to_img(vis_gt, "vis/{}_gt".format(idx))

    ### generate normal image
    cam_info_kb = minibatch_l["cam_info_kb_crop"]
    normal_pred = normal_gener(pred_kb_crop.unsqueeze(1), cam_info_kb.K)
    vis_normal_pred = vis_normal(normal_pred)
    vis_normal_pred = uint8_np_from_img_tensor(vis_normal_pred)
    save_np_to_img(vis_normal_pred, "vis/{}_normal".format(idx))


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

# if is_main_process:
#     logging.info('After Epoch{}/{}, {}'.format(epoch, config['solver']['epochs'], metric.get_result_info()))
#     # writer.add_scalar("Train/loss", loss_meter.mean(), epoch)
#     metric.add_scalar(writer, tag='Test', epoch=epoch)