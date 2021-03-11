#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 22:03
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : config.py
"""

from ruamel import yaml
import collections


def load_config(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.RoundTripLoader)
    return config

### https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
### In case we have more than one config files, use this function to deep merge two nested dictionaries. 
def merge_config(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_config(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                a[key] = b[key]
                print("Overwriting {} from {} to {}".format('.'.join(path + [str(key)]), a[key], b[key]) )
                # raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def save_config(path, config):
    with open(path, 'w') as nf:
        yaml.dump(config, nf, Dumper=yaml.RoundTripDumper)


def print_config(config, step=''):
    for k, v in config.items():
        if isinstance(v, collections.OrderedDict):
            new_step = step + '  '
            print(step + k + ':')
            print_config(v, new_step)
        else:
            print(step + k + ':', v)

class Config:

    def __init__(self, defualt_path='./config/default.yaml'):
        with open(defualt_path) as f:
            self.config = yaml.load(f, Loader=yaml.RoundTripLoader)

    def load(self, path):
        with open(path) as f:
            self.config = yaml.load(f, Loader=yaml.RoundTripLoader)

    def save(self, path):
        with open(path, 'w') as nf:
            yaml.dump(self.config, nf, Dumper=yaml.RoundTripDumper)

    def get(self):
        return self.config

    def set(self, config):
        self.config = config
