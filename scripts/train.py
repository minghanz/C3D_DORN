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

warnings.filterwarnings("ignore")
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# sys.path.append(os.path.abspath(".."))
sys.path.insert(0, os.path.abspath(".."))  
# running in parent dir
os.chdir("..")

from dp.metircs.average_meter import AverageMeter
from dp.utils.config import load_config, print_config, merge_config, save_config
from dp.utils.pyt_io import create_summary_writer
from dp.metircs import build_metrics
from dp.core.solver import Solver
from dp.datasets.loader import build_loader
from dp.visualizers import build_visualizer

def validation(is_main_process, niter_test, bar_format, metric, te_loader, solver, epoch, config, visualizer):
    # validation
    if is_main_process:
        pbar = tqdm(range(niter_test), file=sys.stdout, bar_format=bar_format)
    else:
        pbar = range(niter_test)
    metric.reset()
    test_iter = iter(te_loader)
    count_vis = 0
    for idx in pbar:
        t_start = time.time()
        minibatch = test_iter.next()
        filtered_kwargs = solver.parse_kwargs(minibatch)
        # print(filtered_kwargs)
        t_end = time.time()
        io_time = t_end - t_start
        t_start = time.time()
        pred = solver.step_no_grad(**filtered_kwargs)
        t_end = time.time()
        inf_time = t_end - t_start

        t_start = time.time()
        metric.compute_metric(pred, filtered_kwargs)
        t_end = time.time()
        cmp_time = t_end - t_start

        if is_main_process:
            print_str = '[Test] Epoch{}/{}'.format(epoch, config['solver']['epochs']) \
                        + ' Iter{}/{}: '.format(idx + 1, niter_test) \
                        + metric.get_snapshot_info() \
                        + ' IO:%.2f' % io_time \
                        + ' Inf:%.2f' % inf_time \
                        + ' Cmp:%.2f' % cmp_time
            pbar.set_description(print_str, refresh=False)
        """
        visualization for model output and feature maps.
        """
        if is_main_process and idx % 10 == 0 and count_vis < 10:
            visualizer.visualize(minibatch, pred, epoch=epoch, idx=idx//10)
            count_vis += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-p', '--pathcfg', type=str)    # separate the path-related config items in another file
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

    if is_main_process:
        print_config(config)
        save_config(os.path.join(snap_dir, "config.yaml"), config)

    # dataset
    tr_loader, sampler, niter_per_epoch = build_loader(config, True, solver.world_size, solver.distributed)
    te_loader, _, niter_test = build_loader(config, False, solver.world_size, solver.distributed)


    """
        usage: debug
    """
    # niter_per_epoch, niter_test = 200, 20

    loss_meter = AverageMeter()
    metric = build_metrics(config)
    if is_main_process:
        writer = create_summary_writer(snap_dir)
        visualizer = build_visualizer(config, writer)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    print("--------------vanidation sanity check")
    epoch_init = solver.epoch
    solver.after_epoch()
    validation(is_main_process, 5, bar_format, metric, te_loader, solver, epoch_init, config, visualizer)

    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(solver.epoch + 1, config['solver']['epochs'] + 1):
        solver.before_epoch(epoch=epoch)
        if solver.distributed:
            sampler.set_epoch(epoch)

        if is_main_process:
            pbar = tqdm(range(niter_per_epoch), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = range(niter_per_epoch)
        loss_meter.reset()
        train_iter = iter(tr_loader)
        for idx in pbar:
            t_start = time.time()
            minibatch = train_iter.next()
            filtered_kwargs = solver.parse_kwargs(minibatch)
            # print(filtered_kwargs)
            t_end = time.time()
            io_time = t_end - t_start
            t_start = time.time()
            # loss = solver.step(**filtered_kwargs)
            loss, loss_dict = solver.step(**filtered_kwargs)
            loss_dorn = loss_dict["loss_dorn"]
            loss_c3d = loss_dict["loss_c3d"]
            loss_mean = loss_dict["loss_mean"]
            loss_deviation = loss_dict["loss_deviation"]
            loss_nll = loss_dict["loss_nll"]
            t_end = time.time()
            inf_time = t_end - t_start
            loss_meter.update(loss)

            if is_main_process:
                print_str = '[Train] Epoch{}/{}'.format(epoch, config['solver']['epochs']) \
                            + ' Iter{}/{}:'.format(idx + 1, niter_per_epoch) \
                            + ' lr=%.8f' % solver.get_learning_rates()[0] \
                            + ' loss=%.2f' % loss.item() \
                            + '(%.2f)' % loss_meter.mean() \
                            + ' l_dorn=%.2f' % loss_dorn.item() \
                            + ' l_c3d=%.2f' % loss_c3d.item() \
                            + ' l_mean=%.2f' % loss_mean.item() \
                            + ' l_dev=%.2f' % loss_deviation.item() \
                            + ' l_nll=%.2f' % loss_nll.item() \
                            + ' IO:%.2f' % io_time \
                            + ' Inf:%.2f' % inf_time
                pbar.set_description(print_str, refresh=False)

        solver.after_epoch(epoch=epoch)
        if is_main_process:
            snap_name = os.path.join(snap_dir, 'epoch-{}.pth'.format(epoch))
            solver.save_checkpoint(snap_name)
            ### Minghan: delete the last one to save space
            if epoch > 2 and epoch % 5 != 1:
                last_snap_name = os.path.join(snap_dir, 'epoch-{}.pth'.format(epoch-1))
                os.remove(last_snap_name)

        validation(is_main_process, niter_test, bar_format, metric, te_loader, solver, epoch, config, visualizer)
        # # validation
        # if is_main_process:
        #     pbar = tqdm(range(niter_test), file=sys.stdout, bar_format=bar_format)
        # else:
        #     pbar = range(niter_test)
        # metric.reset()
        # test_iter = iter(te_loader)
        # for idx in pbar:
        #     t_start = time.time()
        #     minibatch = test_iter.next()
        #     filtered_kwargs = solver.parse_kwargs(minibatch)
        #     # print(filtered_kwargs)
        #     t_end = time.time()
        #     io_time = t_end - t_start
        #     t_start = time.time()
        #     pred = solver.step_no_grad(**filtered_kwargs)
        #     t_end = time.time()
        #     inf_time = t_end - t_start

        #     t_start = time.time()
        #     metric.compute_metric(pred, filtered_kwargs)
        #     t_end = time.time()
        #     cmp_time = t_end - t_start

        #     if is_main_process:
        #         print_str = '[Test] Epoch{}/{}'.format(epoch, config['solver']['epochs']) \
        #                     + ' Iter{}/{}: '.format(idx + 1, niter_test) \
        #                     + metric.get_snapshot_info() \
        #                     + ' IO:%.2f' % io_time \
        #                     + ' Inf:%.2f' % inf_time \
        #                     + ' Cmp:%.2f' % cmp_time
        #         pbar.set_description(print_str, refresh=False)
        #     """
        #     visualization for model output and feature maps.
        #     """
        #     if is_main_process and idx % 10 == 0:
        #         visualizer.visualize(minibatch, pred, epoch=epoch)

        if is_main_process:
            logging.info('After Epoch{}/{}, {}'.format(epoch, config['solver']['epochs'], metric.get_result_info()))
            writer.add_scalar("Train/loss", loss_meter.mean(), epoch)
            metric.add_scalar(writer, tag='Test', epoch=epoch)

    if is_main_process:
        writer.close()
