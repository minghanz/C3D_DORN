# import c3d 
import numpy as np 
import os

if __name__ == "__main__":
    root = "/home/minghanz/DORN_pytorch/snap_dir/monodepth/Kitti/dorn"
    runs = os.listdir(root)
    for run in runs:
        path = os.path.join(root, run)
        files = os.listdir(path)
        for f in files:
            if f.endswith(".pth") and not f.endswith("14.pth"):
                f_full = os.path.join(path, f)
                print(f_full)
                os.remove(f_full)